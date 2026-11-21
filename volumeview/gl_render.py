"""Interactive OpenGL backend (PyOpenGL + glfw).

Depth testing is always on, so rotating behind the actor reveals points that
the front view occluded -- the same guarantee the software rasterizer makes.
A translucent quad marks the paper screen; pressing D toggles a depth-only
shade so front/back depth can be read directly.

This backend only initializes a GL context when ``run()`` is called, so the
module imports cleanly on headless machines and the software backend can be
used there instead.
"""

from __future__ import annotations

import glfw
import numpy as np

from .camera import OrbitCamera
from .scene import SCREEN_HALF_HEIGHT, SCREEN_HALF_WIDTH, SCREEN_Z

# OpenGL is imported lazily inside init_window so this module (and the rest of
# the package) imports cleanly on headless machines that lack libGL; the
# software renderer is the usable backend there.
GL = None

POINT_VERT = """
#version 330 core
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_color;
uniform mat4 u_vp;
uniform float u_point_scale;
uniform int u_depth_mode;
out vec3 v_color;
out float v_eye_depth;
void main() {
    vec4 eye = u_vp * vec4(a_pos, 1.0);
    gl_Position = eye;
    gl_PointSize = clamp(u_point_scale / max(0.1, -eye.z), 2.0, 14.0);
    v_eye_depth = -eye.z;
    v_color = a_color;
}
"""

POINT_FRAG = """
#version 330 core
in vec3 v_color;
in float v_eye_depth;
uniform int u_depth_mode;
out vec4 frag;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    if (u_depth_mode == 1) {
        float t = clamp((v_eye_depth - 1.5) / 6.0, 0.0, 1.0);
        frag = vec4(vec3(1.0 - t) * 0.92, 1.0);
    } else {
        frag = vec4(v_color, 1.0);
    }
}
"""

SCREEN_VERT = """
#version 330 core
layout(location=0) in vec3 a_pos;
uniform mat4 u_vp;
void main() { gl_Position = u_vp * vec4(a_pos, 1.0); }
"""

SCREEN_FRAG = """
#version 330 core
uniform int u_depth_mode;
out vec4 frag;
void main() {
    if (u_depth_mode == 1) frag = vec4(0.35, 0.35, 0.4, 1.0);
    else frag = vec4(0.55, 0.6, 0.7, 0.28);
}
"""


def _compile(kind: int, source: str) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"shader compile failed: {log}")
    return shader


def _program(vsrc: str, fsrc: str) -> int:
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, _compile(GL.GL_VERTEX_SHADER, vsrc))
    GL.glAttachShader(prog, _compile(GL.GL_FRAGMENT_SHADER, fsrc))
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"program link failed: {log}")
    return prog


class OpenGLRenderer:
    def __init__(self, width: int = 960, height: int = 720,
                 title: str = "Volumetric Performance Player"):
        self.width, self.height = width, height
        self.title = title
        self.camera = OrbitCamera()
        self.depth_mode = False
        self.show_screen = True
        self._window = None
        self._point_vao = None
        self._point_vbo_pos = None
        self._point_vbo_col = None
        self._point_program = None
        self._screen_vao = None
        self._screen_program = None
        self._drag = None
        self.on_key = None  # callback(key_name)
        self.init_error = ""

    # -- glfw setup --------------------------------------------------------
    def init_window(self) -> bool:
        global GL
        try:
            from OpenGL import GL as _GL
        except ImportError as exc:
            self.init_error = (
                f"OpenGL library unavailable ({exc}); use the headless "
                "backend (`volumeview headless`) or run on a machine with GL")
            return False
        GL = _GL
        if not glfw.init():
            self.init_error = "glfw.init() failed: no display or window system"
            return False
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self._window = glfw.create_window(
            self.width, self.height, self.title, None, None)
        if not self._window:
            glfw.terminate()
            self.init_error = "glfw could not create a GL 3.3 context"
            return False
        glfw.make_context_current(self._window)
        glfw.set_framebuffer_size_callback(self._window, self._on_resize)
        glfw.set_mouse_button_callback(self._window, self._on_button)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor)
        glfw.set_scroll_callback(self._window, self._on_scroll)
        glfw.set_key_callback(self._window, self._on_key)
        self._init_gl()
        return True

    def _init_gl(self) -> None:
        self._point_program = _program(POINT_VERT, POINT_FRAG)
        self._screen_program = _program(SCREEN_VERT, SCREEN_FRAG)
        self._point_vao = GL.glGenVertexArrays(1)
        self._point_vbo_pos = GL.glGenBuffers(1)
        self._point_vbo_col = GL.glGenBuffers(1)
        self._screen_vao = GL.glGenVertexArrays(1)
        self._screen_vbo = GL.glGenBuffers(1)

        h, w = SCREEN_HALF_HEIGHT, SCREEN_HALF_WIDTH
        quad = np.array([
            -w, -h, SCREEN_Z,  w, -h, SCREEN_Z,  w, h, SCREEN_Z,
            -w, -h, SCREEN_Z,  w, h, SCREEN_Z,   -w, h, SCREEN_Z,
        ], dtype=np.float32)
        GL.glBindVertexArray(self._screen_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._screen_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad,
                        GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.04, 0.05, 0.07, 1.0)

    # -- frame upload ------------------------------------------------------
    def upload_frame(self, frame) -> None:
        world = np.ascontiguousarray(frame.world_points, dtype=np.float32)
        colors = np.ascontiguousarray(
            frame.colors.astype(np.float32) / 255.0, dtype=np.float32)
        GL.glBindVertexArray(self._point_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._point_vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, world.nbytes, world,
                        GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._point_vbo_col)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors,
                        GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        self._point_count = world.shape[0]

    def draw(self, frame) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        w, h = glfw.get_framebuffer_size(self._window)
        vp = self.camera.view_projection(w / max(h, 1))

        if self.show_screen:
            GL.glUseProgram(self._screen_program)
            loc = GL.glGetUniformLocation(self._screen_program, "u_vp")
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, vp)
            dloc = GL.glGetUniformLocation(self._screen_program,
                                           "u_depth_mode")
            GL.glUniform1i(dloc, 1 if self.depth_mode else 0)
            GL.glBindVertexArray(self._screen_vao)
            # Depth-write off so the translucent paper never hides points.
            GL.glDepthMask(GL.GL_FALSE)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glDepthMask(GL.GL_TRUE)

        if frame is not None:
            self.upload_frame(frame)
            GL.glUseProgram(self._point_program)
            loc = GL.glGetUniformLocation(self._point_program, "u_vp")
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, vp)
            GL.glUniform1f(GL.glGetUniformLocation(
                self._point_program, "u_point_scale"),
                float(self.height) * 0.12)
            GL.glUniform1i(GL.glGetUniformLocation(
                self._point_program, "u_depth_mode"),
                1 if self.depth_mode else 0)
            GL.glBindVertexArray(self._point_vao)
            GL.glDrawArrays(GL.GL_POINTS, 0, self._point_count)

    # -- input -------------------------------------------------------------
    def _on_resize(self, _win, w, h):
        GL.glViewport(0, 0, w, h)
        self.width, self.height = w, h

    def _on_button(self, win, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._drag = glfw.get_cursor_pos(win)
            else:
                self._drag = None

    def _on_cursor(self, win, x, y):
        if self._drag is None:
            return
        x0, y0 = self._drag
        self.camera.drag(x - x0, y - y0)
        self._drag = (x, y)

    def _on_scroll(self, _win, _dx, dy):
        self.camera.zoom(-dy)

    # Arrow keys have no printable name in glfw, so map them explicitly.
    _KEY_NAMES = None

    def _key_name(self, key: int) -> str:
        if OpenGLRenderer._KEY_NAMES is None:
            OpenGLRenderer._KEY_NAMES = {
                glfw.KEY_SPACE: "space",
                glfw.KEY_LEFT: "left",
                glfw.KEY_RIGHT: "right",
                glfw.KEY_UP: "up",
                glfw.KEY_DOWN: "down",
                glfw.KEY_H: "h", glfw.KEY_J: "j", glfw.KEY_L: "l",
                glfw.KEY_D: "d", glfw.KEY_S: "s",
                glfw.KEY_MINUS: "-", glfw.KEY_EQUAL: "=",
            }
        return OpenGLRenderer._KEY_NAMES.get(
            key, (glfw.get_key_name(key, 0) or str(key)).lower())

    def _on_key(self, win, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_D:
            self.depth_mode = not self.depth_mode
        elif key == glfw.KEY_S:
            self.show_screen = not self.show_screen
        elif self.on_key is not None:
            self.on_key(self._key_name(key))

    def should_close(self) -> bool:
        return glfw.window_should_close(self._window)

    def swap(self) -> None:
        glfw.swap_buffers(self._window)
        glfw.poll_events()

    def terminate(self) -> None:
        glfw.destroy_window(self._window)
        glfw.terminate()
