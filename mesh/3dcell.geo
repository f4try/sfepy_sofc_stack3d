// Gmsh project created on Mon May 25 16:22:56 2020
SetFactory("OpenCASCADE");
//+
Point(1) = {-0, -0, 0, 1.0};
//+
Point(2) = {0, 1e-4, 0, 1.0};
//+
Point(3) = {0, -0.0001, 0, 1.0};
//+
Point(4) = {0, -0.0002, 0, 1.0};
//+
Point(6) = {0.001, 0, 0, 1.0};
//+
Point(7) = {0.001, 0.0001, 0, 1.0};
//+
Point(8) = {0.001, -0.0001, 0, 1.0};
//+
Point(9) = {0.001, -0.0002, 0, 1.0};
//+
//+
Point(10) = {0.00025, 0.0001, 0, 1.0};
//+
Point(11) = {0.00075, 0.0001, 0, 1.0};
//+
Point(12) = {0.00025, -0.0002, 0, 1.0};
//+
Point(13) = {0.00075, -0.0002, 0, 1.0};
//+
//+
Point(14) = {0.00025, 1e-4+0.5e-3, 0, 1.0};
//+
Point(15) = {0.00075, 1e-4+0.5e-3, 0, 1.0};
//+
Point(16) = {0.00075, -2e-4-0.5e-3, 0, 1.0};
//+
Point(17) = {0.00025, -2e-4-0.5e-3, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 10};
//+
Line(3) = {10, 14};
//+
Line(4) = {14, 15};
//+
Line(5) = {15, 11};
//+
Line(6) = {11, 7};
//+
Line(7) = {10, 11};
//+
Line(8) = {7, 6};
//+
Line(9) = {6, 1};
//+
Line(10) = {1, 3};
//+
Line(11) = {3, 8};
//+
Line(12) = {6, 8};
//+
Line(13) = {8, 9};
//+
Line(14) = {9, 13};
//+
Line(15) = {13, 12};
//+
Line(16) = {12, 4};
//+
Line(17) = {4, 3};
//+
Line(18) = {12, 17};
//+
Line(19) = {17, 16};
//+
Line(20) = {16, 13};
//+

//+
Curve Loop(1) = {3, 4, 5, -7};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, 7, 6, 8, 9, 1};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {9, 10, 11, -12};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {11, 13, 14, 15, 16, 17};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {15, 18, 19, 20};
//+
Plane Surface(5) = {5};
//+
//+
Transfinite Curve {3, 4, 5, 7, 15, 18, 19, 20} = 11 Using Progression 1;
//+
Transfinite Curve {2, 1, 10, 17, 16, 14, 13, 12, 8, 6} = 6 Using Progression 1;
//+
Transfinite Curve {9, 11} = 21 Using Progression 1;
//+
Transfinite Surface {1};
//+
Transfinite Surface {5} Right;
//+

//+
Transfinite Surface {3} Alternated;
//+

//+
Transfinite Surface {2} = {2, 1, 6, 7} Alternated;
//+
Transfinite Surface {4} = {3, 4, 9, 8} Alternated;
//+
//+
Extrude {0, 0, 10e-3} {
  Surface{1}; Surface{2}; Surface{3}; Surface{4}; Surface{5}; Layers{20}; Recombine;
}
Physical Volume("topchannel") = {1};
//+
Physical Volume("top") = {2};
//+
Physical Volume("middle") = {3};
//+
Physical Volume("bottom") = {4};
//+
Physical Volume("bottomchannel") = {5};
//+
Physical Surface("topinlet") = {10};
//+
Physical Surface("topoutlet") = {1};
//+
Physical Surface("bottominlet") = {5};
//+
Physical Surface("bottomoutlet") = {30};
//+
//+
Physical Surface("topconnector") = {12, 11};
//+
Physical Surface("bottomconnector") = {22, 24};
