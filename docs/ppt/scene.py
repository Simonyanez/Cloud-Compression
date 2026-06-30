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
        title = Text("Adaptive GFT Clustering for Point Cloud Compression", color=C_TEXT, font_size=40).scale_to_fit_width(14)
        author = Text("Simón Yáñez, Eduardo Pavez, Jorge Silva", color=C_BLUE, font_size=24, slant=ITALIC)
        title_group = VGroup(title, author).arrange(DOWN, buff=0.5)
        
        point_cloud_image = ImageMobject("../poster/longdress_thumb.png").scale(1.2)
        intro_text = Text("Compressing 3D Point Cloud Attributes", font_size=36, color=C_TEXT).next_to(point_cloud_image, RIGHT, buff=1)

        self.play(FadeIn(title_group))
        self.wait(2)
        self.play(FadeOut(title_group), FadeIn(point_cloud_image), FadeIn(intro_text))
        self.wait(2)

        # --- SCENE 2: Classical Fourier ---
        self.next_section("Classical FT")
        self.play(FadeOut(point_cloud_image), FadeOut(intro_text))
        ft_text = Text("Classical Fourier Transform Decomposes a Signal into Simple Waves", color=C_TEXT, font_size=32).to_edge(UP)
        noisy_signal = self.get_noisy_signal_graph()
        basis_functions = self.get_basis_functions()
        self.play(Write(ft_text), Create(noisy_signal))
        self.play(ReplacementTransform(noisy_signal.copy(), basis_functions))
        self.wait(3)

        # --- SCENE 3: Graph Fourier Transform ---
        self.next_section("GFT")
        gft_text = Text("GFT Extends This to Unstructured Graph Signals", color=C_BLUE, font_size=32).to_edge(UP)
        gft_viz = self.get_gft_basis_viz()
        self.play(FadeOut(ft_text), ReplacementTransform(VGroup(noisy_signal, basis_functions), gft_viz['graph']))
        self.play(Write(gft_text))
        self.play(LaggedStart(
            ReplacementTransform(gft_viz['graph'].copy(), gft_viz['basis_dc']),
            ReplacementTransform(gft_viz['graph'].copy(), gft_viz['basis_low']),
            ReplacementTransform(gft_viz['graph'].copy(), gft_viz['basis_high']),
            lag_ratio=0.5, run_time=3
        ))
        self.wait(3)
        
        # --- SCENE 4: Adaptive Topology ---
        self.next_section("Adaptive Topology")
        self.play(FadeOut(gft_text), Uncreate(VGroup(gft_viz['graph'], gft_viz['basis_dc'], gft_viz['basis_low'], gft_viz['basis_high'])))
        topology_text = Text("Adaptive Topology Follows the Signal Gradient", color=C_PURPLE, font_size=36).to_edge(UP)
        math_text = self.get_topology_math().scale(0.9).to_edge(RIGHT, buff=0.5)
        graph_viz = self.get_gradient_graph()

        self.play(Write(topology_text))
        self.play(Create(graph_viz["graph"]), Write(math_text))
        self.play(Indicate(graph_viz["sinks"], color=C_RED, scale_factor=1.5))
        self.play(LaggedStart(*[GrowFromCenter(loop) for loop in graph_viz["loops"]], lag_ratio=0.2))
        self.wait(3)

        # --- SCENE 5: Spatial Regularization & Overhead ---
        self.next_section("Spatial Regularization")
        self.play(FadeOut(topology_text), FadeOut(math_text), Uncreate(graph_viz["graph"]), Uncreate(graph_viz["loops"]))
        spatial_text = Text("Spatial Regularization Creates Compressible Label Runs", color=C_ORANGE, font_size=36).to_edge(UP)
        block_grid = self.get_block_grid()
        
        overhead_chart = self.get_overhead_chart()
        self.play(Write(spatial_text))
        self.play(LaggedStart(*[Create(b) for b in block_grid], lag_ratio=0.05))
        
        for _ in range(3): self.play(*[b.animate.set_color(np.random.choice([C_BLUE, C_GREEN, C_PURPLE])) for b in block_grid], run_time=0.2)
        
        self.play(ReplacementTransform(block_grid, overhead_chart["chaotic"]))
        self.wait(1)
        self.play(Transform(overhead_chart["chaotic"], overhead_chart["smooth"]))
        self.wait(2)

        # --- SCENE 6: Breakthrough at B32 ---
        self.next_section("Breakthrough")
        self.play(FadeOut(spatial_text), FadeOut(overhead_chart["chaotic"]))
        rd_curve_group = self.get_rd_curve()
        self.play(Create(rd_curve_group))
        self.play(self.camera.frame.animate.move_to(rd_curve_group.label).scale(0.6))
        self.play(Indicate(rd_curve_group.label, color=C_GREEN, scale_factor=2))
        self.wait(2)

        # --- SCENE 7: Closing ---
        self.next_section("Closing")
        self.play(self.camera.frame.animate.move_to(ORIGIN).scale(1))
        final_text = Text("PCADC: A Framework for High-Fidelity Compression", color=C_TEXT, font_size=36)
        self.play(FadeOut(rd_curve_group), Write(final_text))
        self.wait(2)

    # --- Helper Methods ---
    def get_noisy_signal_graph(self):
        axes = Axes(x_range=[0, 10], y_range=[-2, 2], axis_config={"color": C_BLUE})
        t = np.linspace(0, 10, 100)
        y = 0.8 * np.sin(1.5 * t) + 0.3 * np.cos(4 * t) + 0.2 * np.random.randn(100)
        return axes.plot(lambda x: np.interp(x, t, y), color=C_TEXT).move_to(ORIGIN)

    def get_basis_functions(self):
        group = VGroup()
        for i in range(2):
            axes = Axes(x_range=[0, 10], y_range=[-1.2, 1.2], x_length=5, y_length=2)
            y = np.cos((i + 1) * np.pi * np.linspace(0, 10, 100) / 10)
            graph = axes.plot(lambda x: np.interp(x, np.linspace(0, 10, 100), y), color=C_GREEN)
            group.add(VGroup(axes, graph))
        return group.arrange(DOWN, buff=0.5).next_to(ORIGIN, DOWN, buff=1.0)
    
    def get_gft_basis_viz(self):
        nodes = VGroup(*[Dot([i*0.6-0.9, j*0.6-0.9, 0], radius=0.1) for i in range(4) for j in range(4)])
        luminance = np.random.rand(16)
        for i, node in enumerate(nodes): node.set_color(interpolate_color(BLUE, YELLOW, luminance[i]))
        graph = VGroup(nodes).move_to(LEFT*5)

        basis_dc = nodes.copy().set_color(interpolate_color(BLUE, YELLOW, np.mean(luminance))).move_to(LEFT*0.5)
        basis_low = nodes.copy()
        for i in range(4):
            for j in range(4): basis_low[i*4+j].set_color(interpolate_color(BLUE, YELLOW, i/3))
        basis_low.move_to(RIGHT*2.5)
        basis_high = nodes.copy()
        for i in range(16): basis_high[i].set_color(BLUE if (i//4 + i%4)%2==0 else YELLOW)
        basis_high.move_to(RIGHT*5.5)
        
        return { "graph": graph, "basis_dc": VGroup(basis_dc, Text("DC",font_size=24).next_to(basis_dc,DOWN)),
                 "basis_low": VGroup(basis_low, Text("Low-Freq",font_size=24).next_to(basis_low,DOWN)),
                 "basis_high": VGroup(basis_high, Text("High-Freq",font_size=24).next_to(basis_high,DOWN)) }

    def get_gradient_graph(self):
        vertices = list(range(8)); edges = [(i, (i+1)%8) for i in range(8)] + [(0,4), (1,5), (2,7)]
        luminance = [0.1, 0.2, 0.9, 0.7, 0.3, 0.4, 0.8, 0.15]
        g = Graph(vertices, edges, vertex_config={"radius": 0.2, "stroke_width": 2}).scale(1.2).to_edge(LEFT, buff=1)
        for i, v in enumerate(g.vertices.values()): v.set_color(interpolate_color(ManimColor(C_PURPLE), ManimColor(C_ORANGE), luminance[i]))
        sinks = VGroup(g.vertices[0], g.vertices[7])
        loops = VGroup(*[Circle(radius=0.25, color=C_RED, stroke_width=4).move_to(s.get_center()) for s in sinks])
        return {"graph": g, "sinks": sinks, "loops": loops}
        
    def get_topology_math(self):
        return MathTex(r"\mathbf{L_a} = \mathbf{L_s} + \mathbf{W}_{sl} \ S_j = \sum_i W_{ij}(y_i - y_j)", tex_to_color_map={"L_a": C_PURPLE, "S_j": C_RED}).scale(1.2)

    def get_block_grid(self):
        return VGroup(*[Square(side_length=0.6, fill_opacity=0.7, stroke_width=1) for _ in range(16)]).arrange_in_grid(4, 4, buff=0.1).to_edge(LEFT)
    
    def get_overhead_chart(self):
        chart_chaotic = BarChart([1.0], bar_names=["Entropy (Chaotic)"], y_range=[0, 1.2], bar_colors=[C_RED]).scale(0.7)
        chart_smooth = BarChart([0.2], bar_names=["CABAC (Smoothed)"], y_range=[0, 1.2], bar_colors=[C_GREEN]).scale(0.7)
        return {"chaotic": VGroup(chart_chaotic).to_edge(RIGHT), "smooth": VGroup(chart_smooth).to_edge(RIGHT)}

    def get_rd_curve(self):
        axes = Axes(x_range=[0, 0.3], y_range=[51, 58], x_length=7, y_length=4.5,
                    axis_config={"color": C_TEXT}, x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
                    y_axis_config={"decimal_number_config": {"num_decimal_places": 0}}).add_coordinates().to_edge(DOWN, buff=1)
        base = [[0.148, 53.51], [0.108, 52.8]]; raw = [[0.145, 53.50], [0.105, 52.8]]; total = [[0.147, 53.50], [0.107, 52.8]]
        baseline = axes.plot_line_graph(x_values=[p[0] for p in base], y_values=[p[1] for p in base], line_color=C_RED, vertex_dot_style={"color": C_RED})
        raw_line = axes.plot_line_graph(x_values=[p[0] for p in raw], y_values=[p[1] for p in raw], line_color=C_GREEN, vertex_dot_style={"color": C_GREEN})
        total_line_vdict = axes.plot_line_graph(x_values=[p[0] for p in total], y_values=[p[1] for p in total], line_color=C_GREEN, add_vertex_dots=False)
        total_line_vdict['line_graph'].set_stroke(width=3, opacity=0.7)
        label = Text("Net Gain: -0.56%", font_size=24, color=C_GREEN).move_to(axes.c2p(0.147, 54.0))
        rd_group = VGroup(axes, baseline, raw_line, total_line_vdict, label)
        setattr(rd_group, 'label', label)
        return rd_group

if __name__ == "__main__":
    pass

