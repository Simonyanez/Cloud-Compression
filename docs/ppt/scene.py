from manim import *

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
        # --- SCENE 1: Opening Title (0-5s) ---
        title = Text("Adaptive GFT Clustering for Point Cloud Compression", color=C_TEXT, font_size=48)
        author = Text("Simón Yáñez, Eduardo Pavez, Jorge Silva", color=C_BLUE, font_size=24, slant=ITALIC)
        VGroup(title, author).arrange(DOWN, buff=0.5)
        self.play(FadeIn(title), FadeIn(author, shift=UP))
        self.wait(3)
        self.play(FadeOut(title), FadeOut(author))

        # --- SCENE 2: The Core Idea: GFT (5-15s) ---
        self.next_section("Core Idea")
        problem_text = Text("Uncompressed Point Cloud Signal", color=C_TEXT, font_size=36).to_edge(UP)
        noisy_signal = self.get_noisy_signal_graph()
        gft_text = Text("GFT Basis Functions", color=C_BLUE, font_size=36).next_to(noisy_signal, DOWN, buff=1.0)
        basis_functions = self.get_basis_functions()
        
        self.play(Write(problem_text))
        self.play(Create(noisy_signal))
        self.play(Write(gft_text))
        self.play(LaggedStart(*[Create(b) for b in basis_functions], lag_ratio=0.5))
        self.wait(2)
        
        # --- SCENE 3: Sparsity & Compression (15-25s) ---
        self.next_section("Sparsity")
        sparsity_text = Text("GFT Coefficients (Sparsity)", color=C_GREEN, font_size=36).to_edge(UP)
        coeffs_chart = self.get_coeffs_chart()
        quantized_chart = self.get_coeffs_chart(quantized=True)
        rlgr_icon = Text("RLGR/Entropy Coder", font_size=24, color=C_ORANGE).next_to(quantized_chart, DOWN)
        
        self.play(FadeOut(problem_text), FadeOut(gft_text), FadeOut(basis_functions))
        self.play(ReplacementTransform(noisy_signal, coeffs_chart), Write(sparsity_text))
        self.wait(2)
        self.play(Transform(coeffs_chart, quantized_chart))
        self.play(Write(rlgr_icon))
        self.wait(2)

        # --- SCENE 4: Adaptive Topology (25-38s) ---
        self.next_section("Adaptive Topology")
        self.play(FadeOut(sparsity_text), FadeOut(coeffs_chart), FadeOut(rlgr_icon))
        
        topology_text = Text("Adaptive Topology via Self-Loops", color=C_PURPLE, font_size=36).to_edge(UP)
        graph = self.get_graph_with_sinks()
        self.play(Write(topology_text))
        self.play(Create(graph['graph']))
        self.play(LaggedStart(*[node.animate.set_color(C_RED) for node in graph['sinks']], lag_ratio=0.2))
        self.play(LaggedStart(*[GrowFromCenter(loop) for loop in graph['loops']], lag_ratio=0.2))
        self.wait(3)

        # --- SCENE 5: Spatial Regularization (38-48s) ---
        self.next_section("Spatial Regularization")
        self.play(FadeOut(topology_text), FadeOut(graph['graph']), FadeOut(graph['sinks']), FadeOut(graph['loops']))
        
        spatial_text = Text("Spatial Regularization ($\beta$ Penalty)", color=C_ORANGE, font_size=36).to_edge(UP)
        morton_path = self.get_morton_path()
        self.play(Write(spatial_text), Create(morton_path['path']))
        self.play(LaggedStart(*[Create(b) for b in morton_path['blocks']], lag_ratio=0.1))
        
        # Animate color switching
        for _ in range(3):
            self.play(*[b.animate.set_color(np.random.choice([C_BLUE, C_GREEN, C_PURPLE])) for b in morton_path['blocks']], run_time=0.2)
        
        penalty_icon = Text("Penalty!", color=C_RED).scale(0.8).next_to(morton_path['path'], UP)
        self.play(Write(penalty_icon))
        # Animate stabilization
        self.play(
            morton_path['blocks'][0].animate.set_color(C_GREEN),
            morton_path['blocks'][1].animate.set_color(C_GREEN),
            morton_path['blocks'][2].animate.set_color(C_GREEN),
            morton_path['blocks'][3].animate.set_color(C_BLUE),
            morton_path['blocks'][4].animate.set_color(C_BLUE),
            morton_path['blocks'][5].animate.set_color(C_BLUE),
            morton_path['blocks'][6].animate.set_color(C_BLUE),
        )
        self.wait(2)

        # --- SCENE 6: Breakthrough at B32 (48-58s) ---
        self.next_section("Breakthrough")
        self.play(FadeOut(spatial_text), FadeOut(morton_path['path']), FadeOut(morton_path['blocks']), FadeOut(penalty_icon))
        
        b32_text = Text("Breakthrough: Compression at B32", color=C_TEXT, font_size=36).to_edge(UP)
        rd_curve_vgroup = self.get_rd_curve()
        label = rd_curve_vgroup[-1]  # The label is the last element in the VGroup
        
        self.play(Write(b32_text))
        self.play(Create(rd_curve_vgroup))
        self.play(self.camera.frame.animate.move_to(label).scale(0.5))
        self.play(Indicate(label, color=C_GREEN, scale_factor=2))
        self.wait(3)

        # --- SCENE 7: Closing (58-60s) ---
        self.next_section("Closing")
        self.camera.frame.move_to(ORIGIN).scale(2)
        final_text = Text("PCADC: A New Frontier for High-Fidelity Compression", color=C_TEXT, font_size=40)
        self.play(FadeOut(b32_text), FadeOut(rd_curve_vgroup))
        self.play(Write(final_text))
        self.wait(2)

    def get_noisy_signal_graph(self):
        axes = Axes(x_range=[0, 10], y_range=[-2, 2], axis_config={"color": C_BLUE})
        t = np.linspace(0, 10, 100)
        y = 0.5 * np.sin(2 * t) + 0.2 * np.cos(5 * t) + 0.3 * np.random.randn(100)
        return axes.plot(lambda x: np.interp(x, t, y), color=C_TEXT).move_to(ORIGIN)

    def get_basis_functions(self):
        group = VGroup()
        for i in range(3):
            axes = Axes(x_range=[0, 10], y_range=[-1, 1], x_length=4, y_length=1.5)
            y = np.cos((i + 1) * np.pi * np.linspace(0, 10, 100) / 10)
            graph = axes.plot(lambda x: np.interp(x, np.linspace(0, 10, 100), y), color=C_GREEN)
            group.add(VGroup(axes, graph))
        return group.arrange(RIGHT, buff=0.5).next_to(ORIGIN, DOWN, buff=1.5)

    def get_coeffs_chart(self, quantized=False):
        values = [1.0, 0.8, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03]
        if quantized:
            values = [1.0, 0.8, 0.2, 0, 0, 0, 0, 0]
        chart = BarChart(values, bar_names=[f"C{i}" for i in range(len(values))], y_range=[0, 1.2],
                         bar_colors=[C_GREEN, C_GREEN, C_GREEN, C_ORANGE, C_ORANGE, C_ORANGE, C_ORANGE, C_ORANGE])
        return chart.scale(0.8).move_to(ORIGIN)
        
    def get_graph_with_sinks(self):
        vertices = list(range(8))
        edges = [(i, (i+1)%8) for i in range(8)] + [(0,4), (1,5)]
        graph = Graph(vertices, edges,
                      vertex_config={"radius": 0.2, "color": C_BLUE}, edge_config={"color": C_TEXT})
        sinks = [graph.vertices[2], graph.vertices[6]]
        loops = [Arc(radius=0.2, start_angle=PI/2, angle=-2*PI).move_to(s.get_center()) for s in sinks]
        return {"graph": graph, "sinks": VGroup(*sinks), "loops": VGroup(*loops)}

    def get_morton_path(self):
        blocks = VGroup(*[Square(side_length=0.5, fill_opacity=0.8) for _ in range(7)]).arrange_in_grid(2, 4, buff=0.1)
        path = VGroup()
        for i in range(len(blocks) - 1):
            path.add(Arrow(blocks[i].get_center(), blocks[i+1].get_center(), buff=0.25, stroke_width=3, color=C_TEXT))
        return {"blocks": blocks, "path": path}

    def get_rd_curve(self):
        axes = Axes(x_range=[0.1, 0.3], y_range=[52, 58], x_length=8, y_length=5,
                    axis_config={"color": C_TEXT}, x_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
                    y_axis_config={"decimal_number_config": {"num_decimal_places": 0}}).add_coordinates()
        axes.to_edge(DOWN)
        
        base_pts = [[0.148, 53.51], [0.108, 52.8], [0.065, 51.7]]
        raw_pts = [[0.145, 53.50], [0.105, 52.8], [0.063, 51.7]]
        total_pts = [[0.147, 53.50], [0.107, 52.8], [0.064, 51.7]]
        
        baseline = axes.plot_line_graph(x_values=[p[0] for p in base_pts], y_values=[p[1] for p in base_pts],
                                        line_color=C_RED, vertex_dot_style={"color": C_RED}, add_vertex_dots=True)
        adaptive_raw = axes.plot_line_graph(x_values=[p[0] for p in raw_pts], y_values=[p[1] for p in raw_pts],
                                            line_color=C_GREEN, vertex_dot_style={"color": C_GREEN}, add_vertex_dots=True)
        adaptive_total_vdict = axes.plot_line_graph(x_values=[p[0] for p in total_pts], y_values=[p[1] for p in total_pts],
                                              line_color=C_GREEN, add_vertex_dots=False)
        # Access the line object within the VDict, which is keyed by 'line_graph'
        adaptive_total_line = adaptive_total_vdict['line_graph']
        adaptive_total_line.set_stroke(width=5, opacity=0.5)

        label = Text("-0.56%", font_size=24, color=C_GREEN).move_to(axes.c2p(0.147, 54.0))
        return VGroup(axes, baseline, adaptive_raw, adaptive_total_vdict, label)

if __name__ == "__main__":
    # To render, run: manim -pql scene.py PCADC
    pass

