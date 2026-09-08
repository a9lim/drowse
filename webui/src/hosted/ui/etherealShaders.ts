export const fullscreenVertex = `#version 300 es
precision highp float;
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`

const field = `
uniform vec2 uResolution;
uniform vec3 uCamera;
uniform float uFocal;
uniform float uRadius;
uniform float uPhase;
uniform float uTouch;
const float TAU = 6.28318530718;
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float hash(vec2 p) {
  vec3 q = fract(vec3(p.xyx) * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
vec3 palette(float t) {
  vec3 c = mix(vec3(0.055, 0.18, 0.28), vec3(0.443, 0.573, 0.945), smoothstep(0.0, 0.32, t));
  c = mix(c, vec3(1.0, 0.63, 0.57), smoothstep(0.23, 0.69, t));
  return mix(c, vec3(1.0, 0.992, 0.91), smoothstep(0.58, 0.94, t));
}
`

const lightField = `
uniform sampler2D uInitial;
uniform float uVideoBlend;
vec3 lightAt(vec2 p) {
  if (uVideoBlend >= 1.0) return texture(uVideo, p).rgb;
  return mix(texture(uInitial, p).rgb, texture(uVideo, p).rgb, smoothstep(0.0, 1.0, uVideoBlend));
}
`

export const sourceFragment = `#version 300 es
precision highp float;
uniform sampler2D uVideo;
in vec2 uv;
out vec4 fragColor;
${field}
${lightField}
void main() {
  vec2 screen = uv * 2.0 - 1.0;
  screen.x *= uResolution.x / uResolution.y;
  vec3 ray = normalize(vec3(screen, -uFocal));
  float b = dot(uCamera, ray);
  float discriminant = b * b - dot(uCamera, uCamera) + uRadius * uRadius;
  vec3 color = vec3(0.006, 0.011, 0.022);
  if (discriminant > 0.0) {
    float nearHit = max(0.04, -b - sqrt(discriminant));
    float farHit = -b + sqrt(discriminant);
    float stepSize = max(0.0, farHit - nearHit) / 56.0;
    float transmission = 1.0;
    for (int i = 0; i < 56; i++) {
      float distance = nearHit + (float(i) + 0.5) * stepSize;
      vec3 p = uCamera + ray * distance;
      float angle = atan(p.z, p.x);
      float radius = length(p);
      float fold = 0.18 * sin(angle * 3.0 + uPhase) * cos(p.y * 3.0 - uPhase);
      float shell = exp(-abs(radius - 1.12 - fold) * 28.0);
      float ribbon = exp(-pow(p.y - 0.3 * sin(angle * 2.0 + uPhase), 2.0) * 85.0)
        * exp(-pow(length(p.xz) - 1.0, 2.0) * 22.0);
      float veil = exp(-abs(p.x * 0.7 + p.z * 0.5 - 0.3 * sin(p.y * 5.0 + uPhase)) * 30.0)
        * (1.0 - smoothstep(0.6, 1.3, radius));
      // The reference supplies light in world space, never a screen-space zoom.
      vec3 light = lightAt(p.xy * vec2(0.27, 0.38) + vec2(0.5) + p.z * vec2(0.04, -0.06));
      float energy = smoothstep(0.12, 0.7, luma(light));
      float density = (shell * 0.8 + ribbon * 1.5 + veil * 0.55) * (0.28 + energy);
      density *= 1.0 - smoothstep(1.4, uRadius, radius);
      float opacity = 1.0 - exp(-density * stepSize * 1.4);
      vec3 emission = mix(palette(0.26 + energy * 0.55), light, 0.25);
      color += transmission * opacity * emission * 2.2;
      transmission *= 1.0 - opacity;
    }
  }
  fragColor = vec4(color, 1.0);
}`

export const particleVertex = `#version 300 es
precision highp float;
uniform sampler2D uVideo;
uniform vec2 uGrid;
uniform float uPixelRatio;
out vec3 vColor;
out float vAlpha;
${field}
${lightField}
void main() {
  float id = float(gl_VertexID);
  vec2 cell = vec2(mod(id, uGrid.x), floor(id / uGrid.x));
  vec2 seed = cell + 0.5;
  float rnd = hash(seed);
  float angle = rnd * TAU;
  float vertical = hash(seed + 31.7) * 2.0 - 1.0;
  float radius = 0.35 + 1.05 * pow(hash(seed + 9.2), 0.3333);
  vec3 p = radius * vec3(sqrt(1.0 - vertical * vertical) * cos(angle), vertical,
    sqrt(1.0 - vertical * vertical) * sin(angle));
  p += 0.025 * vec3(sin(p.y * 6.0 + uPhase), cos(p.z * 5.0 - uPhase), sin(p.x * 4.0 + uPhase));
  vec3 src = lightAt(p.xy * vec2(0.27, 0.38) + vec2(0.5) + p.z * vec2(0.04, -0.06));
  float density = smoothstep(0.12, 0.64, luma(src));
  vec3 relative = p - uCamera;
  float depth = -relative.z;
  if (depth <= 0.06) {
    vAlpha = 0.0; vColor = vec3(0.0);
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0;
    return;
  }
  float glint = pow(0.5 + 0.5 * sin(uPhase * 2.0 + rnd * TAU), 6.0);
  vColor = mix(palette(density), vec3(0.70, 0.84, 1.0), 0.22);
  vAlpha = (0.06 + density * density * 0.4) * (0.4 + glint * 0.3)
    * smoothstep(0.06, 0.35, depth) * exp(-depth * 0.12);
  gl_Position = vec4(relative.x * uFocal / (uResolution.x / uResolution.y), relative.y * uFocal, 0.0, depth);
  gl_PointSize = min(12.0, uPixelRatio * (0.7 + density + glint * 0.5) * 3.0 / depth);
}`

export const particleFragment = `#version 300 es
precision mediump float;
in vec3 vColor;
in float vAlpha;
out vec4 fragColor;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float d = dot(p, p);
  float alpha = (1.0 - smoothstep(0.06, 1.0, d)) * vAlpha;
  fragColor = vec4(vColor * alpha, alpha);
}`

export const blurFragment = `#version 300 es
precision highp float;
uniform sampler2D uInput;
uniform vec2 uStep;
uniform float uTrail;
in vec2 uv;
out vec4 fragColor;
void main() {
  vec3 sum = texture(uInput, uv).rgb * 0.227027;
  sum += texture(uInput, uv + uStep * 1.384615).rgb * 0.316216;
  sum += texture(uInput, uv - uStep * 1.384615).rgb * 0.316216;
  sum += texture(uInput, uv + uStep * 3.230769).rgb * 0.070270;
  sum += texture(uInput, uv - uStep * 3.230769).rgb * 0.070270;
  vec3 trail = texture(uInput, uv - uStep * 5.5).rgb * 0.5;
  trail += texture(uInput, uv - uStep * 9.0).rgb * 0.3;
  trail += texture(uInput, uv - uStep * 14.0).rgb * 0.2;
  fragColor = vec4(mix(sum, max(sum, trail * 0.78), uTrail), 1.0);
}`

export const compositeFragment = `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform sampler2D uGlow;
uniform float uLightTheme;
uniform float uOrbBoost;
in vec2 uv;
out vec4 fragColor;
${field}
vec3 screen(vec3 a, vec3 b) { return 1.0 - (1.0 - a) * (1.0 - b); }
void main() {
  vec2 centered = uv - 0.5;
  vec2 p = centered * (1.0 + dot(centered, centered) * 0.075) + 0.5;
  vec2 texel = 1.0 / uResolution;
  vec2 split = texel * vec2(1.7 + uTouch * 0.7, 0.35);
  vec3 sharp = texture(uScene, p).rgb;
  vec3 fringe = vec3(texture(uScene, p + split).r, sharp.g, texture(uScene, p - split).b);
  sharp = mix(sharp, fringe, 0.72);
  vec3 bloom = texture(uGlow, p).rgb;
  vec2 spread = texel * 18.0;
  vec3 wide = texture(uGlow, p + vec2(spread.x, spread.y)).rgb;
  wide += texture(uGlow, p + vec2(-spread.x, spread.y)).rgb;
  wide += texture(uGlow, p + vec2(spread.x, -spread.y)).rgb;
  wide += texture(uGlow, p - spread).rgb;
  wide = max(wide * 0.25 - vec3(0.012), 0.0);
  vec3 color = screen(sharp, bloom * 0.96);
  color += wide * vec3(0.28, 0.24, 0.52);
  float y = luma(sharp);
  float mist = max(luma(bloom) - 0.028, 0.0);
  color += vec3(0.23, 0.22, 0.46) * pow(mist, 0.8) * 0.38;
  vec2 cell = (floor(p * uResolution / 8.0) + 0.5) * 8.0 / uResolution;
  vec3 pixelated = texture(uScene, cell).rgb;
  color = mix(color, screen(color, pixelated * 0.3), 0.16 * smoothstep(0.06, 0.5, y));
  float threshold = smoothstep(0.17, 0.62, y + (hash(floor(gl_FragCoord.xy / 2.0)) - 0.5) * 0.08);
  color = screen(color, vec3(0.94, 0.96, 1.0) * threshold * 0.16);
  float scanline = 0.5 + 0.5 * cos(gl_FragCoord.y * 2.0943951);
  float scanDepth = mix(0.22, 0.08, smoothstep(0.18, 0.78, y));
  color *= 1.0 - scanDepth * scanline;
  vec3 phosphor = 0.5 + 0.5 * cos(gl_FragCoord.x * 2.0943951 + vec3(0.0, 2.0943951, 4.1887902));
  color *= 0.91 + 0.09 * phosphor;
  float grain = (hash(floor(gl_FragCoord.xy)) - 0.5) * 0.012;
  color += grain * (0.22 + 0.78 * smoothstep(0.02, 0.6, y));
  color = color / (1.0 + color * 0.24);
  color = clamp(color, 0.0, 1.0);
  if (uLightTheme > 0.5) {
    color = vec3(0.94902, 0.95686, 0.97255) + vec3(
      dot(color, vec3(-0.07087, -0.23840, -0.02407)),
      dot(color, vec3(-0.10255, -0.34497, -0.03483)),
      dot(color, vec3(-0.02418, -0.08134, -0.00821))
    );
  } else {
    color = 0.43 * pow(color, vec3(0.55));
    color *= 1.0 + (uOrbBoost - 1.0) * smoothstep(vec3(0.12), vec3(0.30), color);
  }
  fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}`
