from graphviz import Digraph

def create_resbnvox64net_graph():
    dot = Digraph(format='png',
                  graph_attr={'rankdir': 'TB', 'splines': 'ortho'},
                  node_attr={'shape': 'record', 'style': 'filled', 'fillcolor': 'lightblue'})

    # Input block
    dot.node('input', 'Input (1x64x64x64)')

    # First Conv Block
    dot.node('conv1', 'Conv3d(1 -> 30, k=7, s=2)\nBN3d\nReLU')
    dot.node('res1', 'ResVoxBlock(30)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res2', 'ResVoxBlock(30)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res3', 'ResVoxBlock(30)', shape='ellipse', fillcolor='lightyellow')

    # Second Conv Block
    dot.node('conv2', 'Conv3d(30 -> 60, k=3, s=2)\nBN3d\nReLU')
    dot.node('res4', 'ResVoxBlock(60)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res5', 'ResVoxBlock(60)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res6', 'ResVoxBlock(60)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res7', 'ResVoxBlock(60)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res8', 'ResVoxBlock(60)', shape='ellipse', fillcolor='lightyellow')

    # Third Conv Block
    dot.node('conv3', 'Conv3d(60 -> 120, k=3, s=2)\nBN3d\nReLU')
    dot.node('res9', 'ResVoxBlock(120)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res10', 'ResVoxBlock(120)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res11', 'ResVoxBlock(120)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res12', 'ResVoxBlock(120)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res13', 'ResVoxBlock(120)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res14', 'ResVoxBlock(120)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res15', 'ResVoxBlock(120)', shape='ellipse', fillcolor='lightyellow')
    dot.node('res16', 'ResVoxBlock(120)', shape='ellipse', fillcolor='lightyellow')

    # Pooling
    dot.node('pool', 'AvgPool3d(k=3, s=2)')

    # Fully Connected Layers
    dot.node('fc1', 'Linear(3240 -> 512)\nReLU\nDropout')
    dot.node('fc2', 'Linear(512 -> n_classes)')

    # Connections
    dot.edges([
        ('input', 'conv1'),
        ('conv1', 'res1'),
        ('res1', 'res2'),
        ('res2', 'res3'),
        ('res3', 'conv2'),
        ('conv2', 'res4'),
        ('res4', 'res5'),
        ('res5', 'res6'),
        ('res6', 'res7'),
        ('res7', 'res8'),
        ('res8', 'conv3'),
        ('conv3', 'res9'),
        ('res9', 'res10'),
        ('res10', 'res11'),
        ('res11', 'res12'),
        ('res12', 'res13'),
        ('res13', 'res14'),
        ('res14', 'res15'),
        ('res15', 'res16'),
        ('res16', 'pool'),
        ('pool', 'fc1'),
        ('fc1', 'fc2')
    ])

    return dot

# Generate and render the graph
graph = create_resbnvox64net_graph()
graph.render('resbnvox64net_architecture', view=True)
