import { fullscreenVertex, sourceFragment, particleVertex, particleFragment, blurFragment, compositeFragment } from "./etherealShaders";
import { HERO_FOCAL_LENGTH, HERO_VOLUME_RADIUS, heroCameraPosition } from "./heroCamera";

type Program = { program: WebGLProgram; uniforms: Record<string, WebGLUniformLocation | null> };
type Target = { texture: WebGLTexture; framebuffer: WebGLFramebuffer; width: number; height: number };

export function createHeroShaderSource(canvas: HTMLCanvasElement, video: HTMLVideoElement, poster: HTMLImageElement) {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, stencil: false, powerPreference: "low-power" });
  if (!gl) throw new Error("Background rendering unavailable");
  const programs: WebGLProgram[] = [], textures: WebGLTexture[] = [], buffers: WebGLFramebuffer[] = [], shaders: WebGLShader[] = [];
  let vao: WebGLVertexArrayObject | null = null;
  const dispose = () => {
    for (const p of programs) gl.deleteProgram(p);
    for (const t of textures) gl.deleteTexture(t);
    for (const b of buffers) gl.deleteFramebuffer(b);
    for (const s of shaders) gl.deleteShader(s);
    gl.deleteVertexArray(vao);
  };
  const compile = (vertex: string, fragment: string): Program => {
    const program = gl.createProgram()!;
    programs.push(program);
    for (const [type, source] of [[gl.VERTEX_SHADER, vertex], [gl.FRAGMENT_SHADER, fragment]] as const) {
      const shader = gl.createShader(type)!;
      shaders.push(shader);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "Shader compilation failed");
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "Shader linking failed");
    const uniforms: Program["uniforms"] = {};
    for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS); i++) {
      const { name } = gl.getActiveUniform(program, i)!;
      uniforms[name] = gl.getUniformLocation(program, name);
    }
    return { program, uniforms };
  };
  const texture = () => {
    const tex = gl.createTexture()!;
    textures.push(tex);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  };
  const target = (): Target => {
    const tex = texture(), framebuffer = gl.createFramebuffer()!;
    buffers.push(framebuffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { texture: tex, framebuffer, width: 0, height: 0 };
  };
  try {
    const source = compile(fullscreenVertex, sourceFragment), particles = compile(particleVertex, particleFragment);
    const blur = compile(fullscreenVertex, blurFragment), composite = compile(fullscreenVertex, compositeFragment);
    const videoTexture = texture(), initialTexture = texture(), scene = target(), horizontal = target(), glow = target();
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE); gl.disable(gl.DITHER);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    let width = 1, height = 1, pixelRatio = 1, mediaTime = -1, videoWidth = 0, videoHeight = 0;
    let travel = 0, elapsed = 0, entrance = 0, videoBlend = 0, rendered = false;
    const sizeTarget = (t: Target, w: number, h: number) => {
      t.width = w; t.height = h;
      gl.bindTexture(gl.TEXTURE_2D, t.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.framebuffer);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("Background buffers unavailable");
    };
    const bind = (program: Program, destination: Target | null) => {
      gl.useProgram(program.program);
      gl.bindFramebuffer(gl.FRAMEBUFFER, destination?.framebuffer ?? null);
      gl.viewport(0, 0, destination?.width ?? width, destination?.height ?? height);
      return program.uniforms;
    };
    const sample = (u: Program["uniforms"], name: string, tex: WebGLTexture, unit: number) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(u[name], unit);
    };
    const field = (u: Program["uniforms"]) => {
      const mobile = canvas.clientWidth <= 760;
      const camera = heroCameraPosition(travel, width / height, mobile, entrance / 1.6);
      gl.uniform2f(u.uResolution, width, height);
      gl.uniform3f(u.uCamera, ...camera);
      gl.uniform1f(u.uFocal, HERO_FOCAL_LENGTH);
      gl.uniform1f(u.uRadius, HERO_VOLUME_RADIUS);
      gl.uniform1f(u.uPhase, elapsed / 24 * Math.PI * 2);
      gl.uniform1f(u.uTouch, 0.45);
    };
    return {
      resize(w: number, h: number) {
        pixelRatio = Math.min(devicePixelRatio || 1, 1.25, Math.sqrt((w <= 760 ? 360_000 : 800_000) / Math.max(1, w * h)));
        const nextWidth = Math.max(2, Math.round(w * pixelRatio)), nextHeight = Math.max(2, Math.round(h * pixelRatio));
        if (width === nextWidth && height === nextHeight) return;
        width = canvas.width = nextWidth; height = canvas.height = nextHeight;
        sizeTarget(scene, width, height);
        sizeTarget(horizontal, Math.ceil(width / 4), Math.ceil(height / 4));
        sizeTarget(glow, horizontal.width, horizontal.height);
      },
      update(progress: number, delta: number, lightTheme: boolean, orbBoost: number) {
        if (gl.isContextLost()) return false;
        const playingVideo = video.readyState >= 2;
        if (!playingVideo && !(poster.complete && poster.naturalWidth > 0)) return false;
        const lightSource = playingVideo ? video : poster;
        const sourceTime = playingVideo ? video.currentTime : -2;
        const sourceWidth = playingVideo ? video.videoWidth : poster.naturalWidth;
        const sourceHeight = playingVideo ? video.videoHeight : poster.naturalHeight;
        if (!rendered) {
          travel = progress;
          gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, initialTexture);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, lightSource);
          rendered = true;
        }
        elapsed = (elapsed + delta) % 24;
        entrance = Math.min(1.6, entrance + delta);
        if (playingVideo && entrance >= 1.6) videoBlend = Math.min(1, videoBlend + delta);
        const ease = 1 - Math.exp(-delta * 12);
        travel += (progress - travel) * ease;
        if (mediaTime !== sourceTime && (playingVideo || mediaTime < 0)) {
          gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, videoTexture);
          if (videoWidth !== sourceWidth || videoHeight !== sourceHeight) {
            videoWidth = sourceWidth; videoHeight = sourceHeight;
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, videoWidth, videoHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
          }
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, lightSource);
          mediaTime = sourceTime;
        }
        gl.disable(gl.BLEND);
        let u = bind(source, scene);
        field(u); sample(u, "uVideo", videoTexture, 0);
        sample(u, "uInitial", initialTexture, 1); gl.uniform1f(u.uVideoBlend, videoBlend);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        u = bind(particles, scene);
        field(u); sample(u, "uVideo", videoTexture, 0);
        sample(u, "uInitial", initialTexture, 1); gl.uniform1f(u.uVideoBlend, videoBlend);
        const grid = canvas.clientWidth <= 760 ? [112, 72] : [192, 120];
        gl.uniform2f(u.uGrid, grid[0], grid[1]); gl.uniform1f(u.uPixelRatio, pixelRatio);
        gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
        gl.drawArrays(gl.POINTS, 0, grid[0] * grid[1]); gl.disable(gl.BLEND);
        u = bind(blur, horizontal); sample(u, "uInput", scene.texture, 0);
        gl.uniform2f(u.uStep, 8 / width, 0); gl.uniform1f(u.uTrail, 0.7);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        u = bind(blur, glow); sample(u, "uInput", horizontal.texture, 0);
        gl.uniform2f(u.uStep, 0, 5 / height); gl.uniform1f(u.uTrail, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        u = bind(composite, null);
        field(u); sample(u, "uScene", scene.texture, 0); sample(u, "uGlow", glow.texture, 1);
        gl.uniform1f(u.uLightTheme, lightTheme ? 1 : 0);
        gl.uniform1f(u.uOrbBoost, orbBoost);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        return true;
      },
      dispose,
    };
  } catch (error) { dispose(); throw error; }
}
