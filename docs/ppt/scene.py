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
        # --- SCENE 1: Title ---
        title = Text("Adaptive GFT Clustering for Point Cloud Compression", color=C_TEXT, font_size=40).to_edge(UP)
        author = Text("Simón Yáñez, Eduardo Pavez, Jorge Silva", color=C_BLUE, font_size=24, slant=ITALIC).next_to(title, DOWN)
        self.play(Write(title), FadeIn(author, shift=DOWN))
        self.wait(2)

        # --- SCENE 2: Classical Fourier Transform (New) ---
        self.next_section("Classical FT")
        self.play(FadeOut(title), FadeOut(author))
        ft_text = Text("Classical Fourier Transform: Decomposing Signals", color=C_TEXT, font_size=36).to_edge(UP)
        noisy_signal = self.get_noisy_signal_graph()
        basis_functions = self.get_basis_functions()
        
        self.play(Write(ft_text))
        self.play(Create(noisy_signal))
        self.play(Create(basis_functions))
        self.wait(2)

        # --- SCENE 3: Transition to Graphs ---
        self.next_section("Transition to Graphs")
        graph_text = Text("But how do we find a basis for unstructured data?", color=C_RED, font_size=36).to_edge(UP)
        graph_2d = self.get_2d_grid(color_mode="random")
        self.play(FadeOut(ft_text), ReplacementTransform(VGroup(noisy_signal, basis_functions), graph_2d), Write(graph_text))
        self.wait(2)

        # --- SCENE 4: GFT Sparsity ---
        self.next_section("GFT Sparsity")
        gft_sparsity_text = Text("GFT finds a basis that creates sparsity", color=C_GREEN, font_size=36).to_edge(UP)
        coeffs_chart = self.get_coeffs_chart()
        quantized_chart = self.get_coeffs_chart(quantized=True)
        binary_code = Text("101101...", font="monospace", color=C_ORANGE).next_to(quantized_chart, DOWN, buff=0.5)

        self.play(FadeOut(graph_text), ReplacementTransform(graph_2d, coeffs_chart), Write(gft_sparsity_text))
        self.wait(1)
        self.play(Transform(coeffs_chart, quantized_chart), run_time=2)
        self.play(ReplacementTransform(quantized_chart.copy(), binary_code))
        self.wait(2)
        
        # --- SCENE 5: Adaptive Topology with Gradient Viz ---
        self.next_section("Adaptive Topology")
        self.play(FadeOut(gft_sparsity_text), FadeOut(coeffs_chart), FadeOut(binary_code))
        topology_text = Text("Adaptive Topology: Following the Signal Gradient", color=C_PURPLE, font_size=36).to_edge(UP)
        math_text = self.get_topology_math().scale(0.6).to_edge(RIGHT, buff=0.5)
        graph_viz = self.get_gradient_graph()

        self.play(Write(topology_text), Write(math_text))
        self.play(Create(graph_viz["graph"]))
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in graph_viz["arrows"]], lag_ratio=0.2))
        self.play(Indicate(graph_viz["sinks"], color=C_RED, scale_factor=1.5))
        self.play(LaggedStart(*[GrowFromCenter(loop) for loop in graph_viz["loops"]], lag_ratio=0.2))
        self.wait(3)

        # --- SCENE 6: Spatial Regularization ---
        self.next_section("Spatial Regularization")
        self.play(FadeOut(topology_text), FadeOut(math_text), Uncreate(graph_viz["graph"]), Uncreate(graph_viz["arrows"]), Uncreate(graph_viz["loops"]))
        spatial_text = Text("Spatial Regularization Reduces Signaling Overhead", color=C_ORANGE, font_size=36).to_edge(UP)
        beta_math = MathTex(r"J_{total} = J_{RD} + \beta \cdot \mathds{1}(L_i \neq L_{i-1})", color=C_TEXT, 
                            tex_template=TexTemplate(preamble=r"\usepackage{dsfont}\usepackage{amsmath}")).next_to(spatial_text, DOWN)
        morton_slice = self.get_morton_slice()
        
        self.play(Write(spatial_text), Write(beta_math), LaggedStart(*[Create(d) for d in morton_slice], lag_ratio=0.1))
        # ... (rest of scenes are the same)
        self.wait(40) # Placeholder for brevity

    # --- Helper Methods ---
    def get_noisy_signal_graph(self):
        axes = Axes(x_range=[0, 10], y_range=[-2, 2], axis_config={"color": C_BLUE})
        t = np.linspace(0, 10, 100)
        y = 0.5 * np.sin(2 * t) + 0.2 * np.cos(5 * t) + 0.3 * np.random.randn(100)
        return axes.plot(lambda x: np.interp(x, t, y), color=C_TEXT).move_to(ORIGIN).scale(0.8)

    def get_basis_functions(self):
        group = VGroup()
        for i in range(3):
            axes = Axes(x_range=[0, 10], y_range=[-1, 1], x_length=4, y_length=1.5)
            y = np.cos((i + 1) * np.pi * np.linspace(0, 10, 100) / 10)
            graph = axes.plot(lambda x: np.interp(x, np.linspace(0, 10, 100), y), color=C_GREEN)
            group.add(VGroup(axes, graph))
        return group.arrange(RIGHT, buff=0.5).next_to(ORIGIN, DOWN, buff=1.5)

    def get_2d_grid(self, color_mode="random"):
        grid = VGroup()
        colors = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE]
        for i in range(5):
            for j in range(5):
                dot = Dot(radius=0.1).move_to([i*0.5-1, j*0.5-1, 0])
                if color_mode == "random": dot.set_color(np.random.choice(colors))
                else: dot.set_color(interpolate_color(C_PURPLE, C_ORANGE, (i+j)/8))
                grid.add(dot)
        return grid.scale(1.5)

    def get_coeffs_chart(self, quantized=False):
        values = [1.0, 0.8, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03]
        if quantized: values = [1.0, 0.8, 0.2, 0, 0, 0, 0, 0]
        return BarChart(values, bar_names=[f"C{i}" for i in range(len(values))], y_range=[0, 1.2],
                        bar_colors=[C_GREEN, C_GREEN, C_GREEN, C_ORANGE, C_ORANGE, C_ORANGE, C_ORANGE, C_ORANGE]).scale(0.6)

    def get_gradient_graph(self):
        vertices = list(range(8))
        edges = [(i, (i+1)%8) for i in range(8)] + [(0,4), (1,5)]
        luminance = [0.1, 0.2, 0.9, 0.7, 0.3, 0.4, 0.8, 0.6]
        
        g = Graph(vertices, edges, vertex_config={"radius": 0.2, "stroke_width": 2})
        for i, v in enumerate(g.vertices.values()):
            v.set_color(interpolate_color(ManimColor(C_PURPLE), ManimColor(C_ORANGE), luminance[i]))
            
        arrows = VGroup()
        for u, v in edges:
            if luminance[v] < luminance[u]:
                arrows.add(Arrow(g.vertices[u], g.vertices[v], buff=0.2, stroke_width=3, max_tip_length_to_length_ratio=0.2))

        sinks = VGroup(g.vertices[0], g.vertices[5])
        loops = VGroup(*[Circle(radius=0.2, color=C_RED, stroke_width=3).move_to(s.get_center()) for s in sinks])
        return {"graph": g, "arrows": arrows, "sinks": sinks, "loops": loops}
        
    def get_topology_math(self):
        return MathTex(r"\mathbf{L_a} = \mathbf{L_s} + \mathbf{W}_{sl} \\ S_j = \sum_i M_{ij} \\ M_{ij} = W_{ij}(y_i - y_j)",
                       tex_to_color_map={"L_a": C_PURPLE, "S_j": C_RED, "M_{ij}": C_BLUE}).scale(0.8)

    def get_morton_slice(self):
        return VGroup(*[Dot3D(point=[i*0.8 - 2.5, np.sin(i*1.5), 0], radius=0.1) for i in range(12)]).set_color_by_gradient(C_BLUE, C_GREEN)

if __name__ == "__main__":
    pass
