#!/usr/bin/env python

from gallo import write, fe

def change_material(grid, elt):
    centroid = grid.centroid(elt.el_id)
    if all(3/8 < p < 5/8 for p in centroid):
        return elt
    elif all(2/8 < p < 6/8 for p in centroid):
        return elt._replace(mat_id=1)
    else:
        return elt


def main():
    grid = fe.FEGrid(
        node_file="test/test_inputs/symmetric-8.node",
        ele_file="test/test_inputs/symmetric-8.ele"
    )
    elts = [change_material(grid, elt) for elt in grid.elts_list]
    write.write_ele("foo.ele", elts)


if __name__ == '__main__':
    main()
