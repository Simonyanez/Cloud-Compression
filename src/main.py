from parameters import *
from itertools import product

class Researcher():
    def __init__(self):
        self.point_cloud =PointCloud()

    def __call__(self, params: ExperimentParameters):
        param_combinations = self._generate_combinations(params)
        self.point_cloud(params.point_cloud_path)
        for bsize in params.block_size:
            self.point_cloud.do_block_partitioning(params.block_size)
        pass

    def _generate_combinations(self, params: ExperimentParameters):
        multiple_params = [
            params.block_size,
            params.self_loop_weight,
            params.self_loop_percentage,
            params.quantization_steps
        ]
        return list(product(*multiple_params))

    def _save_config(self):
        pass

    def _save_data(self):
        pass

    def _save_plots(self):
        pass

        


if __name__ == "__main__":
    from src.objects import *
    from src.visualization import *
    import utils.ply as ply
    from transforms import *

    point_cloud_path = Path("res/longdress_vox10_1051.ply")
    
    point_cloud = PointCloud()
    point_cloud(point_cloud_path)
    point_cloud.do_block_partitioning(bsize=8)
    block = point_cloud.get_block(2400)
    Vblock, Ablock = block.Vblock, block.Ablock
    
    graph = AttributeGraph(Vblock, Ablock,block_fraction=0.01)
    # M = graph._attribute_motion_matrix(Ablock)
    # print(f"This is attribute motion matrix {M}")
    # S = graph._sink_nodes_vector(M)
    # print(f"This is sink vector {S}")
    GFT_processor = GFT()
    GFT_matrix, Coeffs = GFT_processor(graph, block)
    visualizer = Visualizer()
    visualizer(graph, block)
    visualizer.visualize_block()
    visualizer.add_selected_nodes()
    visualizer.display()
    print(min(GFT_matrix[:,0]), max(GFT_matrix[:,0]))
    visualizer.visualize_base(GFT_matrix[:,0])
    visualizer.add_selected_nodes()
    visualizer.display()
    visualizer.visualize_motion_matrix()
    visualizer.display()
    visualizer.visualize_sink()
    visualizer.display()
    visualizer.visualize_graph()
    visualizer.display()