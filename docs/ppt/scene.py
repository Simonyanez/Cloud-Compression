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
        title = Text("Adaptive GFT Clustering for Point Cloud Compression", color=C_TEXT, font_size=40).scale_to_fit_width(8)
        author = Text("Simón Yáñez, Eduardo Pavez, Jorge Silva", color=C_BLUE, font_size=24, slant=ITALIC)
        title_group = VGroup(title, author).arrange(DOWN, buff=0.5).to_edge(RIGHT, buff=1)
        
        point_cloud_image = ImageMobject("../poster/longdress_thumb.png").scale(1.5).to_edge(LEFT, buff=1)
        self.play(FadeIn(point_cloud_image), Write(title), FadeIn(author))
        self.wait(3)

        # --- SCENE 2: Classical Fourier ---
        self.next_section("Classical FT")
        self.play(FadeOut(point_cloud_image), FadeOut(title_group))
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
        self.play(
            FadeOut(ft_text),
            FadeOut(noisy_signal),
            FadeOut(basis_functions),
        )
        self.play(Write(gft_text), Create(gft_viz['graph']))
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
        math_text = self.get_topology_math().scale(1.0).to_edge(RIGHT, buff=0.8)
        graph_viz = self.get_gradient_graph()

        self.play(Write(topology_text), Create(graph_viz["graph"]))
        self.play(Write(math_text))
        self.play(Indicate(graph_viz["sinks"], color=C_RED, scale_factor=1.5))
        self.play(LaggedStart(*[GrowFromCenter(loop) for loop in graph_viz["loops"]], lag_ratio=0.2))
        self.wait(3)

        # --- SCENE 5: Spatial Regularization & Overhead ---
        self.next_section("Spatial Regularization")
        self.play(FadeOut(topology_text), FadeOut(math_text), Uncreate(graph_viz["graph"]), Uncreate(graph_viz["loops"]))
        spatial_text = Text("Spatial Regularization ($\beta$) Creates Compressible Label Runs", color=C_ORANGE, font_size=36).to_edge(UP)
        block_grid = self.get_block_grid()
        
        overhead_chart = self.get_overhead_chart()
        self.play(Write(spatial_text), LaggedStart(*[Create(b) for b in block_grid], lag_ratio=0.05))
        
        # Animate stabilization
        self.play(
            *[block.animate.set_color(C_GREEN) for block in block_grid[0:8]],
            *[block.animate.set_color(C_BLUE) for block in block_grid[8:13]],
            *[block.animate.set_color(C_PURPLE) for block in block_grid[13:16]],
            run_time=1.5
        )
        binary_code = Text("0000000011111222", font="monospace", color=C_TEXT).next_to(block_grid, DOWN)
        self.play(ReplacementTransform(block_grid.copy(), binary_code))
        self.play(ReplacementTransform(VGroup(block_grid, binary_code), overhead_chart))
        self.wait(3)

        # --- SCENE 6: Breakthrough at B32 ---
        self.next_section("Breakthrough")
        self.play(FadeOut(spatial_text), FadeOut(overhead_chart))
        rd_curve_group = self.get_rd_curve()
        self.play(Create(rd_curve_group))
        self.play(self.camera.frame.animate.move_to(rd_curve_group.label).scale(0.6))
        self.play(Indicate(rd_curve_group.label, color=C_GREEN, scale_factor=1.5))
        self.wait(2)

        # --- SCENE 7: Closing ---
        self.next_section("Closing")
        self.play(self.camera.frame.animate.move_to(ORIGIN).scale(1))
        final_text = Text("PCADC: A Framework for High-Fidelity Compression", color=C_TEXT, font_size=36).scale_to_fit_width(14)
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
        return group.arrange(RIGHT, buff=1.0).next_to(ORIGIN, DOWN, buff=0.8)
    
    def get_gft_basis_viz(self):
        nodes = VGroup(*[Dot([i*0.6-0.9, j*0.6-0.9, 0], radius=0.1) for i in range(4) for j in range(4)])
        edges = []
        for i in range(16):
            if (i+1)%4 != 0: edges.append((i, i+1))
            if i < 12: edges.append((i, i+4))
        graph_obj = Graph(list(range(16)), edges)
        for i, node in enumerate(nodes): graph_obj.vertices[i].move_to(node.get_center())

        luminance = np.random.rand(16)
        for i, node in enumerate(graph_obj.vertices.values()): node.set_color(interpolate_color(BLUE, YELLOW, luminance[i]))
        graph = VGroup(graph_obj).move_to(ORIGIN)

        basis_dc = graph_obj.copy().set_color(interpolate_color(BLUE, YELLOW, np.mean(luminance))).scale(0.5).move_to(DOWN*2.5+LEFT*4)
        basis_low = graph_obj.copy().scale(0.5).move_to(DOWN*2.5)
        for i in range(4):
            for j in range(4): basis_low.vertices[i*4+j].set_color(interpolate_color(BLUE, YELLOW, i/3))
        basis_high = graph_obj.copy().scale(0.5).move_to(DOWN*2.5+RIGHT*4)
        for i in range(16): basis_high.vertices[i].set_color(BLUE if (i//4 + i%4)%2==0 else YELLOW)
        
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
        return MathTex(r"\mathbf{L_a} = \mathbf{L_s} + \mathbf{W}_{sl}", r"S_j = \sum_i W_{ij}(y_i - y_j)",
                       tex_to_color_map={"L_a": C_PURPLE, "S_j": C_RED}).arrange(DOWN, buff=0.5, aligned_edge=LEFT)

    def get_block_grid(self):
        return VGroup(*[Square(side_length=0.6, fill_opacity=0.7, stroke_width=1, color=C_BLUE) for _ in range(16)]).arrange_in_grid(4, 4, buff=0.1).to_edge(LEFT)
    
    def get_overhead_chart(self):
        return BarChart([1.0, 0.15], bar_names=["Chaotic", "Smoothed"], y_range=[0, 1.2], y_length=4,
                        bar_colors=[C_RED, C_GREEN]).scale(0.8).to_edge(RIGHT)

    def get_rd_curve(self):
        axes = Axes(x_range=[0, 0.16], y_range=[53, 57], x_length=8, y_length=5,
                    axis_config={"color": C_TEXT}, x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
                    y_axis_config={"decimal_number_config": {"num_decimal_places": 0}}).add_coordinates().to_edge(DOWN, buff=1)
        # REAL DATA
        base = [[0.1436, 56.57], [0.1049, 55.82], [0.0659, 54.84], [0.0463, 54.18]]
        total = [[0.1428, 56.57], [0.1046, 55.82], [0.0659, 54.84], [0.0466, 54.18]]
        
        baseline = axes.plot_line_graph(x_values=[p[0] for p in base], y_values=[p[1] for p in base], line_color=C_RED, vertex_dot_style={"color": C_RED})
        total_line_vdict = axes.plot_line_graph(x_values=[p[0] for p in total], y_values=[p[1] for p in total], line_color=C_GREEN, vertex_dot_style={"color": C_GREEN})
        
        label = Text("Net Gain: -0.56%", font_size=24, color=C_GREEN).move_to(axes.c2p(0.143, 57.0))
        rd_group = VGroup(axes, baseline, total_line_vdict, label)
        setattr(rd_group, 'label', label)
        return rd_group

if __name__ == "__main__":
    pass
