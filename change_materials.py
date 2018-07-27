#!/usr/bin/env python

from gallo import write, fe

def change_material(grid, elt):
    centroid = grid.centroid(elt.el_id)
    if all(-10/4 < p < 10/4 for p in centroid):
        return elt
    elif all(-5 < p < 5 for p in centroid):
        return elt._replace(mat_id=1)
    else:
        return elt


def main():
    grid = fe.FEGrid(
        node_file="test/test_inputs/origin_centered10.node",
        ele_file="test/test_inputs/origin_centered10.ele"
    )
    elts = [change_material(grid, elt) for elt in grid.elts_list]
    write.write_ele("iron-water10.ele", elts)


if __name__ == '__main__':
    main()
