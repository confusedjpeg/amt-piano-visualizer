import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer;
let pianoGroup, particleField, ringSystems = [], noteSprites = [];
let mouseX = 0, mouseY = 0;
let targetRotationX = 0, targetRotationY = 0;
let pulseIntensity = 0;
let clock = new THREE.Clock();

export function initScene(container) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050508);
  scene.fog = new THREE.Fog(0x050508, 20, 50);

  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(6, 3.5, 10);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
  });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);

  // Lights
  const ambient = new THREE.AmbientLight(0x222244, 0.4);
  scene.add(ambient);

  const goldLight = new THREE.PointLight(0xc9a84c, 2.5, 25);
  goldLight.position.set(3, 5, 4);
  scene.add(goldLight);

  const fillLight = new THREE.DirectionalLight(0x4466aa, 0.6);
  fillLight.position.set(-4, 2, -3);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffeedd, 0.3);
  rimLight.position.set(-2, 1, 6);
  scene.add(rimLight);

  // Ground glow
  const groundGlow = new THREE.Mesh(
    new THREE.PlaneGeometry(30, 30),
    new THREE.MeshBasicMaterial({
      color: 0x0a0a14,
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide,
    })
  );
  groundGlow.rotation.x = -Math.PI / 2;
  groundGlow.position.y = -1.2;
  scene.add(groundGlow);

  buildPiano();
  buildParticles();
  buildRings();
  buildNotes();

  // Mouse tracking
  document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseY = (e.clientY / window.innerHeight) * 2 - 1;
  });

  window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  animate();
}

export function setPulse(intensity) {
  pulseIntensity = Math.min(intensity, 1);
}

function buildPiano() {
  pianoGroup = new THREE.Group();

  const goldMat = new THREE.MeshPhysicalMaterial({
    color: 0xc9a84c,
    metalness: 0.7,
    roughness: 0.25,
    transparent: true,
    opacity: 0.15,
    wireframe: false,
    envMapIntensity: 0.4,
  });

  const wireMat = new THREE.MeshBasicMaterial({
    color: 0xc9a84c,
    wireframe: true,
    transparent: true,
    opacity: 0.12,
  });

  const glowMat = new THREE.MeshPhysicalMaterial({
    color: 0xc9a84c,
    emissive: 0xc9a84c,
    emissiveIntensity: 0.04,
    transparent: true,
    opacity: 0.06,
    roughness: 0.1,
    metalness: 0.9,
  });

  // Piano body - curved grand shape using a custom geometry
  const bodyShape = new THREE.Shape();
  bodyShape.moveTo(0, 0);
  bodyShape.quadraticCurveTo(0.3, 1.8, 1.8, 2.4);
  bodyShape.quadraticCurveTo(3.0, 2.6, 3.8, 2.2);
  bodyShape.quadraticCurveTo(4.2, 2.0, 4.0, 1.6);
  bodyShape.quadraticCurveTo(3.6, 1.2, 2.8, 0.8);
  bodyShape.quadraticCurveTo(2.0, 0.4, 1.2, 0.2);
  bodyShape.quadraticCurveTo(0.6, 0.05, 0, 0);

  const extrudeSettings = {
    steps: 1,
    depth: 0.15,
    bevelEnabled: true,
    bevelThickness: 0.05,
    bevelSize: 0.03,
    bevelSegments: 6,
  };
  const bodyGeo = new THREE.ExtrudeGeometry(bodyShape, extrudeSettings);
  bodyGeo.translate(-1.5, -1.0, -0.075);
  const body = new THREE.Mesh(bodyGeo, glowMat);
  body.position.set(0, 0, 0);
  pianoGroup.add(body);

  // Lid wireframe
  const lidShape = new THREE.Shape();
  lidShape.moveTo(0.1, 0.1);
  lidShape.quadraticCurveTo(0.4, 1.7, 1.7, 2.2);
  lidShape.quadraticCurveTo(2.8, 2.4, 3.5, 2.0);
  lidShape.quadraticCurveTo(3.8, 1.8, 3.6, 1.5);
  lidShape.quadraticCurveTo(3.2, 1.1, 2.5, 0.7);
  lidShape.quadraticCurveTo(1.8, 0.4, 1.0, 0.2);
  lidShape.quadraticCurveTo(0.5, 0.1, 0.1, 0.1);

  const lidGeo = new THREE.ExtrudeGeometry(lidShape, {
    steps: 1,
    depth: 0.02,
    bevelEnabled: false,
  });
  lidGeo.translate(-1.5, -0.9, 0.12);
  const lid = new THREE.Mesh(lidGeo, wireMat);
  pianoGroup.add(lid);

  // Keyboard - row of white keys
  const keyMat = new THREE.MeshPhysicalMaterial({
    color: 0xf0ebe3,
    metalness: 0.1,
    roughness: 0.5,
    transparent: true,
    opacity: 0.08,
  });
  const keyMatBlack = new THREE.MeshPhysicalMaterial({
    color: 0x1a1a24,
    metalness: 0.3,
    roughness: 0.4,
    transparent: true,
    opacity: 0.6,
  });

  const startX = -0.4;
  const keyWidth = 0.08;
  const keyGap = 0.005;
  const totalKeys = 32;

  for (let i = 0; i < totalKeys; i++) {
    const isBlack = [1, 3, 6, 8, 10].includes(i % 12);
    const kw = isBlack ? 0.05 : keyWidth;
    const kh = isBlack ? 0.3 : 0.5;
    const ky = isBlack ? 0.05 : 0;
    const kx = startX + i * (keyWidth + keyGap);

    const keyGeo = new THREE.BoxGeometry(kw, 0.01, kh);
    const key = new THREE.Mesh(keyGeo, isBlack ? keyMatBlack : keyMat);
    key.position.set(kx, 0.01, ky);
    pianoGroup.add(key);
  }

  // Legs
  const legMat = new THREE.MeshPhysicalMaterial({
    color: 0x1a1a24,
    metalness: 0.8,
    roughness: 0.2,
    transparent: true,
    opacity: 0.4,
  });

  const legPositions = [[-0.8, -1.2, -0.8], [1.2, -1.2, -0.8], [0.2, -1.2, 1.0]];
  legPositions.forEach((pos) => {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.06, 0.4, 8), legMat);
    leg.position.set(pos[0], -0.8, pos[2]);
    pianoGroup.add(leg);
  });

  // Subtle glow ring under piano
  const ringGeo = new THREE.RingGeometry(1.5, 2.2, 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0xc9a84c,
    transparent: true,
    opacity: 0.03,
    side: THREE.DoubleSide,
  });
  const glowRing = new THREE.Mesh(ringGeo, ringMat);
  glowRing.rotation.x = -Math.PI / 2;
  glowRing.position.y = -1.0;
  pianoGroup.add(glowRing);

  pianoGroup.position.y = 0.5;
  scene.add(pianoGroup);
}

function buildParticles() {
  const count = 3000;
  const positions = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const speeds = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos((Math.random() * 2) - 1);
    const r = 3 + Math.random() * 12;
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = (Math.random() - 0.3) * 5;
    positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    sizes[i] = 0.01 + Math.random() * 0.04;
    speeds[i] = 0.1 + Math.random() * 0.4;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  geo.setAttribute('speed', new THREE.BufferAttribute(speeds, 1));

  const mat = new THREE.PointsMaterial({
    color: 0xc9a84c,
    size: 0.025,
    transparent: true,
    opacity: 0.4,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });

  particleField = new THREE.Points(geo, mat);
  scene.add(particleField);
}

function buildRings() {
  const ringCount = 6;
  for (let i = 0; i < ringCount; i++) {
    const radius = 1.8 + i * 0.9;
    const geo = new THREE.RingGeometry(radius, radius + 0.008, 64);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xc9a84c,
      transparent: true,
      opacity: 0.03 + Math.random() * 0.04,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(geo, mat);
    ring.rotation.x = -Math.PI / 2 + (Math.random() - 0.5) * 0.2;
    ring.position.y = -0.8 + Math.random() * 0.3;
    ring.userData = {
      speed: 0.1 + Math.random() * 0.2,
      phase: Math.random() * Math.PI * 2,
    };
    scene.add(ring);
    ringSystems.push(ring);
  }
}

function buildNotes() {
  const noteShapes = ['𝄞', '♩', '♪', '♫', '♬'];
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 64;
  canvas.height = 64;

  for (let i = 0; i < 8; i++) {
    const char = noteShapes[i % noteShapes.length];
    ctx.clearRect(0, 0, 64, 64);
    ctx.fillStyle = 'rgba(201, 168, 76, 0.08)';
    ctx.font = '36px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(char, 32, 34);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const sprite = new THREE.Sprite(mat);
    const theta = Math.random() * Math.PI * 2;
    const r = 2 + Math.random() * 5;
    sprite.position.set(
      r * Math.cos(theta),
      -1 + Math.random() * 4,
      r * Math.sin(theta)
    );
    sprite.scale.set(0.3 + Math.random() * 0.4, 0.3 + Math.random() * 0.4, 1);
    sprite.userData = {
      theta: theta,
      r: r,
      speed: 0.05 + Math.random() * 0.1,
      floatOffset: Math.random() * Math.PI * 2,
      floatSpeed: 0.2 + Math.random() * 0.3,
    };
    scene.add(sprite);
    noteSprites.push(sprite);
  }
}

function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  // Camera orbit
  targetRotationY += (mouseX * 0.3 - targetRotationY) * 0.02;
  targetRotationX += (-mouseY * 0.15 - targetRotationX) * 0.02;

  const radius = 11;
  const baseAngle = t * 0.04;
  camera.position.x = radius * Math.sin(baseAngle + targetRotationY * 0.5);
  camera.position.z = radius * Math.cos(baseAngle + targetRotationY * 0.5);
  camera.position.y = 3 + targetRotationX * 1.5;
  camera.lookAt(0, 0, 0);

  // Piano hover
  if (pianoGroup) {
    pianoGroup.position.y = 0.5 + Math.sin(t * 0.3) * 0.04;
    pianoGroup.rotation.y = Math.sin(t * 0.15) * 0.02;
  }

  // Particles
  if (particleField) {
    const positions = particleField.geometry.attributes.position.array;
    const speeds = particleField.geometry.attributes.speed.array;
    const pulse = pulseIntensity * 2;

    for (let i = 0; i < positions.length / 3; i++) {
      positions[i * 3 + 1] += Math.sin(t * speeds[i] + i) * 0.001 * (1 + pulse);
      const angle = t * 0.02 * speeds[i];
      const x = positions[i * 3];
      const z = positions[i * 3 + 2];
      const newX = x * Math.cos(angle * 0.001) - z * Math.sin(angle * 0.001);
      const newZ = x * Math.sin(angle * 0.001) + z * Math.cos(angle * 0.001);
      positions[i * 3] = newX;
      positions[i * 3 + 2] = newZ;
    }
    particleField.geometry.attributes.position.needsUpdate = true;
    particleField.material.opacity = 0.3 + pulse * 0.3;
    particleField.material.size = 0.025 + pulse * 0.02;
  }

  // Rings
  ringSystems.forEach((ring, i) => {
    const { speed, phase } = ring.userData;
    ring.scale.setScalar(1 + Math.sin(t * speed + phase) * 0.05);
    ring.material.opacity = (0.03 + Math.sin(t * speed * 0.5 + phase) * 0.02) * (1 + pulseIntensity);
    ring.rotation.z = Math.sin(t * speed * 0.3 + phase) * 0.02;
  });

  // Note sprites
  noteSprites.forEach((sprite) => {
    const { theta, r, speed, floatOffset, floatSpeed } = sprite.userData;
    const angle = t * speed;
    sprite.position.x = r * Math.cos(theta + angle);
    sprite.position.z = r * Math.sin(theta + angle);
    sprite.position.y += Math.sin(t * floatSpeed + floatOffset) * 0.002;
    sprite.material.opacity = 0.06 + Math.sin(t * floatSpeed + floatOffset) * 0.03;
  });

  renderer.render(scene, camera);
}

export function resize() {
  if (!renderer) return;
  const container = renderer.domElement.parentElement;
  if (container) {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }
}
