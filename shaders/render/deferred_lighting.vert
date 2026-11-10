#version 460 core
//  shaders/render/deferred_lighting.vert                               [DEV 2]
//  Triangle plein écran généré depuis gl_VertexID (aucun vertex buffer, VAO vide).
//  Sommets : (-1,-1), (3,-1), (-1,3) -> couvre tout [-1, 1]².
//  z = 0 : en Reverse-Z c'est la profondeur "infini" (ciel). Le mode "ciel seul" s'en sert
//  avec glDepthFunc(GL_EQUAL) pour ne peindre que les pixels où rien n'a été dessiné.

out vec2 vUv;

void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
