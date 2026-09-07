import type { BackgroundSettings } from "./appearance";

export const BACKGROUND_VERTEX = `
attribute vec2 position;
varying vec2 uv;
void main() { uv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }
`;

export const BACKGROUND_FRAGMENT = `
precision mediump float;
uniform sampler2D image;
uniform vec2 resolution;
uniform vec2 imageSize;
uniform float pixelSize;
uniform float effect;
varying vec2 uv;
float bayer(vec2 point) {
  vec2 p = mod(floor(point), 4.0);
  vec2 a = mod(p, 2.0);
  vec2 b = floor(p / 2.0);
  return (4.0 * (2.0 * a.x + 3.0 * a.y - 4.0 * a.x * a.y)
    + (2.0 * b.x + 3.0 * b.y - 4.0 * b.x * b.y) + 0.5) / 16.0;
}
void main() {
  vec2 cell = floor(uv * resolution / pixelSize);
  vec2 coord = effect < 0.5 ? uv : (cell + 0.5) * pixelSize / resolution;
  float scale = max(resolution.x / imageSize.x, resolution.y / imageSize.y);
  coord = (coord - 0.5) * resolution / (imageSize * scale) + 0.5;
  vec4 color = texture2D(image, vec2(coord.x, 1.0 - coord.y));
  if (effect > 1.5) color.rgb = floor(color.rgb * 5.0 + bayer(cell)) / 5.0;
  gl_FragColor = color;
}
`;

export function backgroundRenderer(canvas: HTMLCanvasElement) {
  const gl = canvas.getContext("webgl", { alpha: true, antialias: false, depth: false, preserveDrawingBuffer: false });
  if (!gl) return null;
  const shaders: WebGLShader[] = [];
  function compile(type: number, source: string): WebGLShader | null {
    const shader = gl!.createShader(type)!;
    shaders.push(shader);
    gl!.shaderSource(shader, source);
    gl!.compileShader(shader);
    return gl!.getShaderParameter(shader, gl!.COMPILE_STATUS) ? shader : null;
  }
  const vertex = compile(gl.VERTEX_SHADER, BACKGROUND_VERTEX);
  const fragment = compile(gl.FRAGMENT_SHADER, BACKGROUND_FRAGMENT);
  const program = gl.createProgram()!;
  const buffer = gl.createBuffer();
  const texture = gl.createTexture();
  function dispose() {
    gl!.deleteTexture(texture);
    gl!.deleteBuffer(buffer);
    gl!.deleteProgram(program);
    shaders.forEach(shader => gl!.deleteShader(shader));
  }
  if (!vertex || !fragment) { dispose(); return null; }
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) { dispose(); return null; }
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const locations = Object.fromEntries(["resolution", "imageSize", "pixelSize", "effect"].map(name => [name, gl.getUniformLocation(program, name)]));
  return {
    image(image: HTMLImageElement) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.uniform2f(locations.imageSize, image.naturalWidth, image.naturalHeight);
    },
    draw(settings: BackgroundSettings) {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(locations.resolution, canvas.width, canvas.height);
      gl.uniform1f(locations.pixelSize, settings.pixelSize);
      gl.uniform1f(locations.effect, settings.effect === "original" ? 0 : settings.effect === "pixel" ? 1 : 2);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
    dispose,
  };
}
