// Gmsh project created on Thu Jun 04 12:52:57 2020
SetFactory("OpenCASCADE");
//+
Point(1) = {0, -0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {2, -0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
