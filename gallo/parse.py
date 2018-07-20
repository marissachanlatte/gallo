from pathlib import Path

from gallo import fe

oo = float('inf')

def parse_num_lines(line):
    return int(line.split(' ')[0])


def _split_and_remove_whitespace(line):
    return ' '.join(line.split()).split(' ')


def parse_nodes(node_file):
    node_file = Path(node_file)
    assert node_file.exists(), "Node file: {} does not exist.".format(node_file)
    with node_file.open() as nf:
        num_nodes = parse_num_lines(nf.readline())
        nodes = []

        xmin = ymin = oo
        xmax = ymax = -oo
        for line in nf.readlines():
            if line.startswith('#'):
                continue
            node_id, x, y, boundary =  _split_and_remove_whitespace(line)
            x, y = float(x), float(y)
            # Set boundary data
            xmin = min(x, xmin)
            ymin = min(y, ymin)

            xmax = max(x, xmax)
            ymax = max(y, ymax)

            is_interior = not int(boundary)
            nodes.append(fe.Node((x, y), int(node_id), is_interior))
    assert num_nodes == len(nodes)
    return nodes, (xmin, xmax, ymin, ymax)


def parse_elts(ele_file):
    ele_file = Path(ele_file)
    assert ele_file.exists(), "Ele file: {} does not exist.".format(ele_file)
    with ele_file.open() as ef:
        num_elts = parse_num_lines(ef.readline())
        elts_list = []
        for line in ef.readlines():
            if line.startswith('#'):
                continue
            el_id, *vertices, mat_id = _split_and_remove_whitespace(line)
            vertices = tuple(int(vert) for vert in vertices)
            elts_list.append(fe.Element(int(el_id), vertices, int(mat_id)))
    assert num_elts == len(elts_list)
    return elts_list
