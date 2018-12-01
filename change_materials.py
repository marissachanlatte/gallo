#!/usr/bin/env python

import attr
from gallo import write, fe

def change_material(grid, elt):
    centroid = grid.centroid(elt.el_id)
    if all(-4 < p < 4 for p in centroid):
        return attr.evolve(elt,mat_id=1)
    else:
        return elt


def main():
    grid = fe.FEGrid(
        node_file="test/test_inputs/origin_centered10.node",
        ele_file="test/test_inputs/origin_centered10.ele"
    )
    elts = [change_material(grid, elt) for elt in grid.elts_list]
    write.write_ele("mod-uo2.ele", elts)


if __name__ == '__main__':
    main()
