#!/usr/bin/env python
"""
First solve the stationary electric conduction problem. Then use its
results to solve the evolutionary heat conduction problem.

Run this example as on a command line::

    $ python <path_to_this_file>/thermal_electric.py
"""
from __future__ import absolute_import
from sys import path
# import numpy as nm
# from sfepy.base.conf import ProblemConf
# from sfepy.examples.multi_physics.biot_parallel_interactive import solve_problem
import imageio
from traitsui.group import Group
from traitsui.item import Item
from traitsui.menu import ModalButtons, OKCancelButtons
from traitsui.view import View

import simple
import postproc
# import params
# from var_TPBR import i_c

path.append( '.' )
from os import system
# from var_TPBL import i_a
# import sfepy.discrete.fem.periodic as per
from traits.api import HasTraits, Str, Int,Float
# from sfepy import data_dir
# from mayavi import mlab
# import re

def run():
    system("python simple.py sofc_sfepy_data3d.py")
def post():
    system("python postproc.py output/sofc2d_mesh.10.vtk  -b --wireframe")
    # for i in range(10):
    #     system(f"python postproc.py output/sofc2d_mesh.0{i}.vtk  -o output/result.0{i}.png -n --wireframe")
def mov():
    frames = []
    for i in range(10):
        frames.append(imageio.imread(f"output/result.0{i}.png"))
    imageio.mimsave("output/result.gif", frames, 'GIF', duration=0.5)
    # system("ffmpeg - r 10 - sameq - i output/result. % *.png output/result.mov")
    system("ffmpeg -y -r 2 -f image2 -i output/result.%2d.png -vcodec libx264 output/result.mp4")
class Index(HasTraits):
    V_cell = Float(0.7)
    electric_conductivity = Float(1.5)
    help_txt = Str("修改参数，点击Ok按钮进行计算")
    about_txt = Str("2020 燃料电池有限元计算软件")
view1 = View(
    Group(
        Group(
            Item(name='V_cell', label=u"电池电压(V):", tooltip=u"输入电池电压0.2~1.8V"),
            Item(name='electric_conductivity', label=u"电导率(S):", tooltip=u"输入电池电极的电导率"),
            label=u'电池参数',
            # show_border=True,
            show_labels = True
        ),
        Group(
            Item('help_txt',style='readonly',label='帮助'),
            label=u'帮助',
            # show_border=True,
            show_labels = False
        ),
        Group(
            Item('about_txt',style='readonly',label='关于'),
            label=u'关于',
            # show_border=True,
            show_labels = False
        ),
        orientation = 'horizontal',layout='tabbed'
    ),
    title = u"燃料电池仿真", resizable = True
    ,buttons=OKCancelButtons
)
if __name__ == '__main__':
    index = Index()
    index.configure_traits(view=view1)
    print(index.V_cell)
    data = ''
    with open('sofc_sfepy_data3d.py', 'r+') as f:
        for line in f.readlines():
            # print(line)
            if (line.find("#V_cell")>=0):
                print(line)
                line ="\t'left' : ('Left', {'T.0' : 0.0, 'phis.0' :"+ str(index.V_cell)+"}), #V_cell000"+ '\n'
                print(line)
            if (line.find("#electric_conductivity")>=0):
                print(line)
                line ="\t\t'electric_conductivity' :" + str(index.electric_conductivity)+", #electric_conductivity000"+ '\n'
                print(line)
            data += line
    with open('sofc_sfepy_data3d.py', 'r+') as f:
        f.writelines(data)
    f.close()
# def run():
#     os.system("python simple.py sofc_sfepy_data3d.py")
# def post():
#     os.system("python postproc.py output/sofc2d_mesh.vtk  -b --wireframe")
# if __name__ == '__main__':
#     simple.main()
#     postproc.main()
    run()
    post()
#     mov()