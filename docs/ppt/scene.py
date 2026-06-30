from manim import *
import numpy as np

# Presentation theme colors
C_BACKGROUND = "#1D1F21"
C_TEXT = "#C5C8C6"
C_BLUE = "#5F8787"
C_GREEN = "#8C9440"
C_RED = "#A54242"
C_PURPLE = "#85678F"
C_ORANGE = "#DE935F"

config.background_color = C_BACKGROUND
config.frame_width = 16
config.frame_height = 9

class PCADC(MovingCameraScene):
    def construct(self):
        # --- SCENE 1: Title & Context ---
        title = Text("Adaptive GFT Clustering for Point Cloud Compression", color=C_TEXT, font_size=48)
        author = Text("Simón Yáñez, Eduardo Pavez, Jorge Silva", color=C_BLUE, font_size=24, slant=ITALIC)
        title_group = VGroup(title, author).arrange(DOWN, buff=0.5)
        
        point_cloud_image = ImageMobject("../poster/longdress_thumb.png").scale(0.8).to_edge(LEFT, buff=1)
        intro_text = Text("Compressing 3D Point Cloud Attributes", font_size=36, color=C_TEXT).next_to(point_cloud_image, RIGHT, buff=1)

        self.play(FadeIn(title_group))
        self.wait(2)
        self.play(ReplacementTransform(title_group, intro_text), FadeIn(point_cloud_image))
        self.wait(2)

        # --- SCENE 2: Classical Fourier to GFT ---
        self.next_section("FT to GFT")
        self.play(FadeOut(intro_text), FadeOut(point_cloud_image))
        gft_motivation = Text("GFT extends Fourier analysis to unstructured data", color=C_TEXT, font_size=36).to_edge(UP)
        
        # New GFT visualization
        gft_viz = self.get_gft_basis_viz()
        self.play(Write(gft_motivation))
        self.play(Create(gft_viz['graph']))
        self.play(LaggedStart(
            ReplacementTransform(gft_viz['graph'].copy(), gft_viz['basis1']),
            ReplacementTransform(gft_viz['graph'].copy(), gft_viz['basis2']),
            ReplacementTransform(gft_viz['graph'].copy(), gft_viz['basis3']),
            lag_ratio=0.5, run_time=3
        ))
        self.wait(3)

        # --- SCENE 3: Adaptive Topology ---
        self.next_section("Adaptive Topology")
        self.play(FadeOut(gft_motivation), Uncreate(VGroup(gft_viz['graph'], gft_viz['basis1'], gft_viz['basis2'], gft_viz['basis3'])))
        topology_text = Text("Adaptive Topology Follows the Signal Gradient", color=C_PURPLE, font_size=36).to_edge(UP)
        math_text = self.get_topology_math().scale(0.8).to_edge(RIGHT, buff=0.5)
        graph_viz = self.get_gradient_graph()

        self.play(Write(topology_text), Write(math_text))
        self.play(Create(graph_viz["graph"]))
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in graph_viz["arrows"]], lag_ratio=0.2))
        self.play(Indicate(graph_viz["sinks"], color=C_RED, scale_factor=1.5))
        self.play(LaggedStart(*[GrowFromCenter(loop) for loop in graph_viz["loops"]], lag_ratio=0.2))
        self.wait(3)

        # --- SCENE 4: Spatial Regularization ---
        self.next_section("Spatial Regularization")
        self.play(FadeOut(topology_text), FadeOut(math_text), Uncreate(graph_viz["graph"]), Uncreate(graph_viz["arrows"]), Uncreate(graph_viz["loops"]))
        spatial_text = Text("Spatial Regularization Creates Compressible Label Runs", color=C_ORANGE, font_size=36).to_edge(UP)
        beta_math = MathTex(r"J_{total} = J_{RD} + \beta \cdot \mathds{1}(L_i \neq L_{i-1})", color=C_TEXT, tex_template=TexTemplate(preamble=r"\usepackage{dsfont}\usepackage{amsmath}")).next_to(spatial_text, DOWN)
        
        block_grid = self.get_block_grid()
        self.play(Write(spatial_text), Write(beta_math))
        self.play(LaggedStart(*[Create(b) for b in block_grid], lag_ratio=0.05))
        
        # Animate chaotic switching
        for _ in range(3):
            self.play(*[b.animate.set_color(np.random.choice([C_BLUE, C_GREEN, C_PURPLE])) for b in block_grid], run_time=0.2)
        
        # Animate stabilization into runs
        self.play(
            *[block.animate.set_color(C_GREEN) for block in block_grid[0:8]],
            *[block.animate.set_color(C_BLUE) for block in block_grid[8:13]],
            *[block.animate.set_color(C_PURPLE) for block in block_grid[13:16]],
            run_time=2
        )
        self.wait(2)
        
        # --- SCENE 5 & 6: Breakthrough & Closing ---
        # (Keeping these scenes as they were well-received)
        self.wait(20) # Placeholder for brevity

    # --- Helper Methods ---
    def get_gft_basis_viz(self):
        # Main graph with luminance values
        nodes = VGroup(*[Dot([i-1.5, j-1.5, 0], radius=0.15) for i in range(4) for j in range(4)])
        luminance = np.random.rand(16)
        for i, node in enumerate(nodes):
            node.set_color(interpolate_color(BLUE, YELLOW, luminance[i]))
        graph = VGroup(nodes).move_to(LEFT*4)
        
        # Basis 1 (DC)
        basis1 = nodes.copy().set_color(interpolate_color(BLUE, YELLOW, np.mean(luminance))).move_to(RIGHT*0)
        basis1_text = Text("DC Basis", font_size=24, color=C_TEXT).next_to(basis1, DOWN)
        
        # Basis 2 (Low Freq)
        basis2 = nodes.copy()
        for i in range(4):
            for j in range(4):
                basis2[i*4+j].set_color(interpolate_color(BLUE, YELLOW, i/3))
        basis2.move_to(RIGHT*4)
        basis2_text = Text("Low-Freq Basis", font_size=24, color=C_TEXT).next_to(basis2, DOWN)

        # Basis 3 (High Freq)
        basis3 = nodes.copy()
        for i in range(16):
            basis3[i].set_color(BLUE if i%2 == 0 else YELLOW)
        basis3.move_to(RIGHT*8) # This will be off-screen initially
        
        return {
            "graph": graph,
            "basis1": VGroup(basis1, basis1_text),
            "basis2": VGroup(basis2, basis2_text),
            "basis3": basis3,
        }

    def get_gradient_graph(self):
        vertices = list(range(8))
        edges = [(i, (i+1)%8) for i in range(8)] + [(0,4), (1,5), (2,7)]
        luminance = [0.1, 0.2, 0.9, 0.7, 0.3, 0.4, 0.8, 0.15]
        
        g = Graph(vertices, edges, vertex_config={"radius": 0.2, "stroke_width": 2})
        for i, v in enumerate(g.vertices.values()):
            v.set_color(interpolate_color(ManimColor(C_PURPLE), ManimColor(C_ORANGE), luminance[i]))
            
        arrows = VGroup()
        for u, v_idx in edges:
            if luminance[v_idx] < luminance[u]:
                arrows.add(Arrow(g.vertices[u], g.vertices[v_idx], buff=0.2, stroke_width=4, max_tip_length_to_length_ratio=0.2, color=WHITE))

        sinks = VGroup(g.vertices[0], g.vertices[7])
        loops = VGroup(*[Circle(radius=0.25, color=C_RED, stroke_width=4).move_to(s.get_center()) for s in sinks])
        return {"graph": g, "arrows": arrows, "sinks": sinks, "loops": loops}
        
    def get_topology_math(self):
        return MathTex(r"\mathbf{L_a} = \mathbf{L_s} + \mathbf{W}_{sl} \ S_j = \sum_i W_{ij}(y_i - y_j)",
                       tex_to_color_map={"L_a": C_PURPLE, "S_j": C_RED}).scale(1.0)

    def get_block_grid(self):
        return VGroup(*[Square(side_length=0.5, fill_opacity=0.7, stroke_width=1) for _ in range(16)]).arrange_in_grid(4, 4, buff=0.1)

if __name__ == "__main__":
    pass

