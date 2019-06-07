const fs = require('fs');

var d=[[146,817,699,421],[38,66,83,87],[45,31,31,53],[45,2,46,56],[46,51,16,50],[51,46,50,17],[8,88,93,47],[35,49,41,34],[38,52,42,47],[45,50,42,39],[45,51,42,45],[49,50,42,61],[50,51,42,58],[49,53,42,48],[45,52,42,53],[46,50,42,55],[51,51,42,69],[52,51,42,65],[51,50,42,75],[41,50,42,73],[39,51,42,74],[46,52,42,47],[38,51,42,67],[54,52,42,63],[40,50,42,57],[36,50,42,42],[55,51,42,65],[48,53,42,42],[55,50,42,61],[50,51,42,76],[49,50,42,75],[54,50,42,59],[38,51,42,68],[44,51,42,75],[51,50,42,77],[40,52,42,48],[51,52,42,50],[51,52,42,64],[41,53,42,42],[43,53,42,74],[48,52,42,60],[48,51,42,37],[45,51,42,41],[39,50,42,62],[45,51,42,36],[54,50,42,76],[51,51,42,72],[41,52,42,58],[54,50,42,69],[53,51,42,46],[48,50,42,39],[51,52,42,71],[36,51,42,65],[49,50,42,62],[36,50,42,68],[47,50,42,77],[55,50,42,60],[37,51,42,46],[50,53,42,40],[51,52,42,44],[52,51,42,60],[39,50,42,69],[39,51,42,75],[40,52,42,63],[36,50,42,61],[47,50,42,44],[54,50,42,66],[43,50,42,61],[53,50,42,39],[42,51,42,37],[38,51,42,58],[55,50,42,71],[44,52,42,79],[40,51,42,50],[55,50,42,74],[38,55,42,41],[38,51,42,66],[52,50,42,78],[41,52,42,51],[36,52,42,64],[46,52,42,50],[50,52,42,63],[37,52,42,43],[48,54,42,41],[43,52,42,66],[46,50,42,42],[40,52,42,71],[53,50,42,74],[45,50,42,46],[38,50,42,76],[49,51,42,68],[55,50,42,70],[43,50,42,49],[48,53,42,67],[48,52,42,66],[45,50,42,35],[42,51,42,46],[38,51,42,77],[49,51,42,65],[46,51,42,59],[46,51,42,69],[42,58,42,54],[47,52,42,43],[46,51,42,65],[36,51,42,75],[47,53,42,49],[45,50,42,61],[43,50,42,39],[48,52,42,78],[42,51,42,38],[44,51,42,78],[45,50,42,49],[53,52,42,75],[49,52,42,59],[40,50,42,78],[42,50,42,78],[36,51,42,71],[44,52,42,48],[40,52,42,64],[53,51,42,58],[42,50,42,77],[56,50,42,66],[49,53,42,45],[43,51,42,60],[41,50,42,62],[48,51,42,79],[55,50,42,62],[40,53,42,56],[48,52,42,74],[56,50,42,39],[44,50,42,50],[45,50,42,37],[42,53,42,68],[38,50,42,59],[45,52,42,72],[37,50,42,73],[52,50,42,57],[55,51,42,64],[47,50,42,58],[46,50,42,60],[41,51,42,76],[39,50,42,73],[48,54,42,56],[45,52,42,51],[49,51,42,73],[44,51,42,44],[45,50,42,38],[40,51,42,79],[36,50,42,70],[45,51,42,62],[40,51,42,65],[44,54,42,57],[52,52,42,43],[37,50,42,62],[43,51,42,73],[47,52,42,46],[49,50,42,64],[43,51,42,47],[45,52,42,63],[39,51,42,60],[42,54,42,67],[45,52,42,64],[45,51,42,52],[36,52,42,63],[43,51,42,69],[43,51,42,65],[48,51,42,38],[42,52,42,59],[45,53,42,70],[45,50,42,77],[41,51,42,72],[46,51,42,76],[42,50,42,55],[43,51,42,43],[53,51,42,68],[45,50,42,58],[45,51,42,40],[39,52,42,44],[45,51,42,56],[53,53,42,42],[56,50,42,40],[45,53,42,71],[52,51,42,47],[50,50,42,55],[54,52,42,41],[36,50,42,74],[53,51,42,77],[40,52,42,70],[40,52,42,45],[53,50,42,61],[51,51,42,79],[38,50,42,61],[51,52,42,70],[50,50,42,51],[36,50,42,67],[53,51,42,73],[49,50,42,69],[36,52,42,72],[36,51,42,60],[53,52,42,67],[100,151,69,77],[58,4,6,30],[74,5,101,33],[59,9,148,33],[66,5,119,57],[68,5,15,69],[62,5,280,61],[75,5,174,53],[78,5,211,75],[67,5,169,69],[76,5,209,73],[71,7,221,54],[72,5,70,38],[63,5,232,58],[62,5,285,70],[78,5,76,64],[64,6,215,42],[75,5,12,61],[73,5,144,52],[78,5,76,43],[78,5,34,63],[79,5,83,60],[75,6,95,64],[79,5,41,43],[70,5,142,49],[70,5,142,33],[72,5,185,46],[66,5,168,42],[61,5,133,68],[60,6,249,62],[76,5,8,38],[66,5,21,56],[72,5,171,35],[74,5,239,46],[74,5,145,75],[74,5,173,45],[61,5,198,58],[75,5,174,71],[69,5,141,53],[64,5,137,35],[60,5,21,56],[75,5,102,42],[78,5,76,45],[78,5,211,72],[77,5,75,33],[70,5,142,50],[74,5,188,55],[77,5,115,39],[67,5,80,54],[69,5,16,66],[64,5,283,70],[79,5,41,51],[71,6,229,75],[67,5,151,41],[68,5,203,42],[75,5,174,72],[79,5,29,37],[75,5,208,50],[69,5,28,63],[59,7,181,41],[67,5,48,73],[70,5,53,31],[70,5,73,60],[65,5,303,72],[74,5,268,73],[66,5,139,72],[70,5,142,47],[70,5,125,34],[71,5,87,36],[62,5,295,73],[64,5,288,74],[66,5,185,46],[78,5,76,33],[76,6,24,61],[67,5,77,31],[69,5,26,73],[68,5,15,59],[64,5,251,73],[64,5,167,46],[75,5,102,35],[60,6,11,60],[59,11,125,34],[61,5,270,64],[77,5,241,55],[69,5,105,34],[66,6,157,51],[69,7,46,56],[61,5,269,54],[78,5,76,53],[76,5,162,71],[77,5,70,38],[73,5,144,34],[60,5,165,55],[78,5,159,65],[73,5,225,66],[75,5,174,74],[76,6,218,47],[63,5,307,75],[72,5,65,39],[67,5,169,42],[77,5,183,44],[77,5,75,34],[68,5,72,31],[63,5,294,70],[63,5,232,53],[72,5,47,70],[69,5,128,40],[77,6,195,71],[70,5,142,44],[59,13,19,48],[74,6,97,67],[63,5,232,54],[68,5,31,51],[74,5,173,43],[71,5,149,34],[78,5,76,41],[71,5,8,38],[79,5,24,61],[73,5,152,42],[66,5,51,31],[63,5,232,55],[69,5,16,48],[78,5,76,40],[68,5,15,49],[74,5,153,44],[68,5,15,61],[64,5,283,69],[62,5,108,63],[67,5,86,75],[77,5,176,50],[72,5,113,69],[69,5,16,47],[74,5,101,64],[76,5,21,56],[67,5,86,67],[73,5,144,73],[63,5,21,56],[68,5,15,73],[68,5,39,54],[59,12,42,32],[67,5,29,37],[76,5,103,34],[67,5,47,70],[78,5,69,38],[64,6,11,60],[66,5,247,54],[76,5,89,59],[68,5,15,74],[69,5,16,68],[62,5,261,52],[63,5,200,57],[65,5,189,56],[67,5,17,35],[62,6,214,41],[76,5,103,33],[79,5,41,45],[59,5,264,54],[69,5,43,36],[66,5,188,55],[71,5,108,63],[60,5,165,65],[64,5,254,43],[69,5,28,51],[69,5,10,59],[73,5,10,59],[77,5,176,43],[73,5,182,43],[67,5,169,55],[67,5,155,49],[70,5,142,45],[60,5,12,61],[72,5,74,34],[67,5,38,36],[78,5,64,36],[65,5,201,50],[72,5,222,58],[78,5,76,54],[63,5,232,43],[62,5,288,74],[59,6,212,37],[75,5,58,31],[69,5,45,54],[66,5,116,65],[79,5,41,74],[68,5,15,36],[59,8,185,46],[77,5,210,75],[60,5,242,39],[78,5,76,47],[74,5,101,72],[64,5,225,66],[68,5,203,52],[60,5,290,74],[76,7,197,74],[77,5,210,52],[77,5,146,65],[65,5,234,46],[79,5,41,47],[74,5,101,34],[61,5,291,70],[79,5,41,65],[79,5,41,66],[75,5,122,57],[65,5,70,38],[64,5,292,75],[75,5,174,58],[69,5,28,46],[69,5,28,72],[64,5,254,58],[76,5,89,58],[74,5,101,71],[79,5,41,73],[65,5,129,57],[70,5,142,43],[63,5,232,44],[79,5,41,52],[71,5,152,42],[61,5,304,73],[66,5,12,61],[61,5,85,36],[74,5,114,39],[61,5,260,43],[69,5,28,43],[59,9,77,31],[77,5,82,58],[78,5,76,46],[61,5,270,67],[61,5,270,63],[65,7,33,62],[71,5,236,52],[71,5,205,58],[73,5,227,72],[78,5,211,69],[73,6,177,35],[66,5,119,35],[59,5,289,73],[79,5,41,53],[62,5,253,39],[79,5,41,32],[67,5,133,68],[59,5,264,42],[67,5,169,50],[72,5,143,71],[73,5,127,36],[60,5,165,63],[71,5,205,53],[61,6,297,75],[70,6,71,59],[63,5,286,74],[65,5,126,35],[65,5,27,58],[72,5,218,47],[63,5,232,46],[78,5,18,37],[66,5,119,60],[59,5,298,69],[68,5,15,66],[74,5,101,57],[79,5,41,46],[73,5,88,58],[60,5,23,59],[61,5,303,72],[65,5,138,55],[61,5,156,50],[70,5,121,74],[71,5,112,57],[68,5,15,68],[71,5,170,43],[73,5,88,64],[76,5,175,70],[65,5,138,73],[68,5,182,43],[68,5,15,33],[71,5,250,69],[79,5,41,41],[69,5,28,38],[72,5,110,72],[73,5,238,69],[70,5,121,53],[65,5,293,74],[71,5,170,35],[64,5,14,39],[64,7,84,64],[74,5,101,62],[59,5,271,55],[74,5,57,31],[71,7,83,60],[70,5,204,73],[65,5,10,59],[59,5,230,65],[73,5,88,55],[59,10,157,51],[77,5,210,67],[65,5,179,39],[75,6,29,37],[59,5,224,61],[66,5,202,47],[60,5,165,57],[70,5,117,69],[66,5,255,52],[73,5,238,57],[77,5,106,41],[73,5,206,46],[62,5,244,43],[68,5,15,65],[78,5,76,44],[76,6,187,53],[78,5,61,31],[70,5,142,46],[69,5,118,74],[76,5,103,35],[73,5,21,56],[70,5,121,54],[75,7,184,45],[67,5,169,46],[67,5,86,74],[70,5,142,39],[69,5,77,31],[63,5,284,69],[67,5,32,58],[62,6,160,67],[76,5,103,40],[66,5,202,59],[68,5,15,71],[79,5,44,39],[74,7,92,36],[59,6,92,36],[75,5,208,47],[66,6,153,44],[72,5,113,43],[76,5,240,52],[62,5,231,62],[74,6,46,56],[75,5,102,40],[68,5,15,57],[64,5,8,38],[63,5,306,73],[76,5,159,65],[77,5,123,36],[79,5,41,67],[59,5,191,63],[67,5,169,38],[68,5,15,34],[66,5,255,58],[78,5,61,62],[69,5,16,67],[66,5,139,71],[70,5,178,38],[64,5,80,54],[74,5,71,59],[72,5,55,31],[70,5,121,75],[70,5,180,40],[77,6,46,56],[62,5,84,64],[66,5,119,33],[63,6,161,68],[69,5,28,69],[72,5,131,65],[77,5,10,59],[77,5,190,62],[70,14,19,48],[70,5,121,70],[62,5,299,72],[79,5,41,33],[78,5,76,58],[62,5,253,57],[64,5,276,57],[77,5,210,66],[62,5,277,68],[76,5,209,67],[76,6,83,60],[71,5,79,39],[64,5,167,41],[76,5,209,69],[59,6,30,50],[70,5,121,65],[59,12,20,49],[73,6,178,38],[68,5,39,47],[75,5,185,46],[67,5,86,66],[69,5,28,50],[67,5,83,60],[72,5,113,42],[59,5,294,70],[68,6,11,60],[66,5,202,37],[61,5,198,61],[70,6,95,64],[59,5,275,57],[76,5,209,66],[78,5,76,51],[75,5,122,69],[73,5,172,44],[65,5,138,33],[67,5,235,53],[76,5,186,50],[73,5,144,33],[59,6,244,43],[63,5,232,42],[72,5,183,44],[74,5,101,68],[70,5,32,58],[59,5,230,62],[79,5,94,59],[73,5,88,65],[63,5,166,65],[68,5,15,48],[68,5,15,75],[69,5,28,64],[73,5,150,39],[78,5,76,39],[67,5,140,32],[68,5,186,50],[69,5,141,41],[79,5,41,55],[60,6,302,69],[61,5,198,47],[59,5,230,58],[67,5,86,65],[74,5,145,65],[77,5,13,63],[59,5,130,60],[75,6,108,63],[59,8,245,44],[61,6,248,56],[60,5,161,68],[76,5,40,39],[77,5,123,40],[75,5,102,34],[65,5,201,41],[79,5,41,54],[78,5,71,59],[66,5,168,38],[75,7,221,54],[68,5,15,58],[70,5,204,52],[62,5,267,54],[68,5,39,44],[59,5,192,64],[64,5,137,53],[79,5,82,58],[77,5,210,72],[68,5,15,64],[67,5,169,40],[71,5,256,70],[69,5,12,61],[69,5,28,65],[64,5,49,31],[64,5,167,55],[67,5,163,72],[65,5,201,66],[59,6,213,40],[71,5,162,71],[76,5,175,44],[74,6,220,52],[70,9,106,41],[79,5,62,40],[69,5,28,37],[66,7,13,63],[79,5,41,57],[69,5,28,62],[65,5,138,68],[69,5,28,70],[71,5,170,44],[70,13,42,32],[72,5,237,53],[69,5,141,44],[65,6,182,43],[62,5,225,66],[61,5,198,40],[68,5,15,70],[72,6,184,45],[79,5,41,72],[73,5,56,31],[72,5,113,74],[72,5,237,57],[79,5,62,36],[72,6,13,63],[79,5,41,71],[77,5,7,37],[76,5,240,57],[66,5,119,75],[65,5,201,37],[62,5,199,38],[61,5,266,42],[60,5,252,58],[64,5,292,72],[60,5,265,54],[79,5,91,38],[79,5,62,35],[65,5,138,51],[71,5,87,40],[65,5,138,52],[67,5,169,39],[75,5,183,44],[73,5,144,71],[73,6,136,74],[63,6,213,40],[60,5,290,72],[76,5,175,43],[67,5,35,64],[71,5,99,68],[64,6,287,71],[79,5,104,63],[78,5,76,42],[69,5,16,49],[66,5,118,74],[66,5,246,53],[59,7,287,71],[79,5,41,69],[77,5,196,73],[71,7,7,37],[75,5,174,55],[65,5,138,67],[60,5,290,73],[68,5,39,40],[75,5,174,70],[64,5,233,56],[75,5,116,65],[61,5,270,65],[72,5,109,64],[70,5,142,35],[59,5,298,72],[70,6,37,72],[69,5,141,45],[64,5,25,67],[63,5,273,64],[65,5,193,69],[67,5,124,33],[78,5,211,73],[73,6,187,53],[69,5,17,35],[75,5,102,41],[62,5,278,71],[74,5,219,50],[74,8,157,51],[74,5,207,54],[60,5,290,70],[75,5,79,39],[64,5,14,36],[73,5,144,51],[79,5,78,34],[78,5,211,67],[69,5,28,52],[68,5,15,37],[68,5,15,41],[59,5,274,67],[65,6,92,36],[69,5,28,71],[62,5,281,65],[74,5,145,42],[79,5,41,70],[76,6,152,42],[68,5,185,46],[73,5,206,50],[78,5,12,61],[60,5,252,38],[72,5,74,36],[59,8,246,53],[71,13,20,49],[67,5,22,57],[72,5,143,59],[68,5,15,56],[75,5,93,38],[62,6,223,59],[60,5,290,75],[62,6,92,36],[68,5,15,53],[61,5,270,57],[72,6,133,68],[68,5,70,38],[59,5,298,66],[60,5,165,64],[74,5,18,37],[78,5,76,34],[70,5,31,51],[77,5,134,69],[62,6,130,60],[70,7,96,66],[59,5,223,59],[70,6,188,55],[73,5,88,62],[70,5,98,63],[69,5,28,55],[70,5,121,68],[65,5,234,44],[67,5,169,43],[67,5,140,71],[63,5,232,62],[66,5,202,50],[67,5,220,52],[69,5,22,57],[75,6,164,75],[76,5,240,55],[66,5,202,41],[70,5,29,37],[79,5,41,56],[61,8,7,37],[74,5,101,58],[63,5,286,71],[61,5,270,55],[74,6,96,66],[77,5,241,46],[71,5,154,47],[79,5,41,75],[72,5,237,73],[77,5,75,35],[70,6,24,61],[69,5,141,42],[68,5,15,62],[69,5,148,33],[61,5,10,59],[72,5,100,62],[76,5,217,46],[70,5,73,62],[65,5,111,65],[76,5,59,31],[76,5,89,62],[70,5,142,42],[62,5,262,58],[66,5,139,67],[71,5,107,62],[74,5,134,69],[77,5,90,64],[73,6,128,40],[78,5,76,52],[79,5,41,42],[60,6,301,66],[66,5,119,66],[64,5,137,33],[76,5,209,72],[65,5,234,40],[72,5,74,40],[63,5,300,72],[77,5,81,57],[64,5,249,62],[62,5,253,55],[76,5,103,41],[73,5,144,75],[72,5,220,52],[59,5,258,39],[79,5,91,62],[66,5,134,69],[75,5,174,62],[78,5,211,66],[77,6,194,70],[69,5,52,39],[62,5,243,40],[71,5,228,73],[62,5,253,42],[78,5,11,60],[71,6,148,33],[64,5,282,65],[59,5,289,75],[59,11,216,45],[67,5,154,47],[64,5,23,59],[68,5,188,55],[63,5,272,63],[76,7,133,68],[66,5,139,73],[74,6,11,60],[66,5,180,40],[65,5,201,53],[74,5,218,47],[59,5,275,68],[69,5,28,58],[75,5,182,43],[69,5,147,32],[74,5,34,63],[67,5,169,56],[70,5,92,36],[60,5,160,67],[74,5,263,70],[72,5,12,61],[75,5,158,59],[78,6,30,50],[71,5,54,31],[79,5,41,68],[67,5,125,34],[71,5,99,74],[61,5,296,74],[68,5,15,67],[60,5,259,42],[78,5,63,35],[65,6,135,70],[77,5,60,31],[66,5,202,45],[68,5,15,72],[68,5,15,35],[66,5,119,34],[67,5,169,45],[75,5,174,73],[71,5,170,46],[68,5,179,39],[72,5,9,56],[61,5,67,38],[61,5,65,39],[65,5,201,54],[71,5,99,65],[64,5,279,61],[73,5,226,67],[59,6,218,47],[65,5,305,75],[78,5,76,55],[71,5,205,50],[70,5,53,57],[63,5,178,38],[63,5,132,66],[66,5,168,39],[78,5,76,57],[72,5,219,50],[67,5,66,61],[66,5,139,68],[64,5,245,44],[68,5,203,45],[69,5,120,75],[70,5,195,71],[63,6,220,52],[73,6,24,61],[79,5,41,31],[79,5,84,64],[59,5,289,74],[59,7,220,52],[62,7,219,50],[59,9,177,35],[65,5,50,31],[70,7,97,67],[63,5,232,39],[71,6,157,51],[64,6,108,63],[68,5,39,32],[73,5,206,47],[71,5,170,45],[59,5,248,56],[62,5,299,69],[75,5,102,33],[62,8,218,47],[63,5,12,61],[73,5,36,70],[75,5,122,68],[79,5,41,44],[72,5,100,55],[65,5,50,61],[67,5,68,59],[59,5,257,38]];

var H = [
],
A = d[0][0] % d[0][3] * (d[0][0] % d[0][3]) + d[0][1] % d[0][3] * 2 + d[0][2] % d[0][3],
D = d[1][0] % d[1][3] + d[1][1] % d[1][3] - d[1][2] % d[1][3],
C = d[2][0] % d[2][3] + d[2][1] % d[2][3] - d[2][2] % d[2][3],
Aa = d[3][0] % d[3][3] + d[3][1] % d[3][3] - d[3][2] % d[3][3],
Ba = 0,
N = [
],
L = [
],
Ca = 0,
O = [
],
M = [
],
E = [
],
F = [
],
I = 0,
za = 0,
R = !1,
S = - 1,
T = - 1,
Da = 3,
G = 1,
P = 1
for (x = 5; x < Aa + 5; x++) {
  var Ea = d[x][0] - d[4][1],
  Fa = d[x][1] - d[4][0],
  Ga = d[x][2] - d[4][3];
  z = d[x][3] - Ea - d[4][2];
  H[x - 5] = [
    (Ea + 256).toString(16).substring(1) + ((Fa + 256 << 8) + Ga).toString(16).substring(1),
    z
  ]
}
for (w = 0; w < C; w++) for (E[w] = [
], F[w] = [
], v = 0; v < D; v++) E[w][v] = 0,
F[w][v] = 0;
var V = Aa + 5,
Ha = d[V][0] % d[V][3] * (d[V][0] % d[V][3]) + d[V][1] % d[V][3] * 2 + d[V][2] % d[V][3],
Ia = d[V + 1];
for (x = V + 2; x <= V + 1 + Ha; x++) for (v = d[x][0] - Ia[0] - 1; v < d[x][0] - Ia[0] + d[x][1] - Ia[1] - 1; v++) E[d[x][3] - Ia[3] - 1][v] = d[x][2] - Ia[2];
var Ja = !1,
W = Aa + 7 + Ha;
if (d.length > W) {
  Ja = !0;
  var Ka = [
  ];
  for (w = 0; w < C; w++) for (Ka[w] = [
  ], v = 0; v < D; v++) Ka[w][v] = 0;
  var La = d[W][0] % d[W][3] * (d[W][0] % d[W][3]) + d[W][1] % d[W][3] * 2 + d[W][2] % d[W][3],
  Ma = d[W + 1];
  for (x = W + 2; x <= W + 1 + La; x++) for (v = d[x][0] - Ma[0] - 1; v < d[x][0] - Ma[0] + d[x][1] - Ma[1] - 1; v++) Ka[d[x][3] - Ma[3] - 1][v] = d[x][2] - Ma[2]
}
for (w = 0; w < C; w++) {
  N[w] = [
  ];
  L[w] = [
  ];
  for (v = 0; v < D; ) {
    var Na = v;
    for (z = E[w][v]; v < D && E[w][v] == z; ) v++;
    0 < v - Na && 0 < z && (N[w][N[w].length] = [
      v - Na,
      z
    ], L[w][N[w].length] = !1)
  }
  N[w].length > Ba && (Ba = N[w].length)
}
for (v = 0; v < D; v++) {
  O[v] = [
  ];
  M[v] = [
  ];
  for (w = 0; w < C; ) {
    var Oa = w;
    for (z = E[w][v]; w < C && E[w][v] == z; ) w++;
    0 < w - Oa && 0 < z && (O[v][O[v].length] = [
      w - Oa,
      z
    ], M[v][O[v].length] = !1)
  }
  O[v].length > Ca && (Ca = O[v].length)
}

var output = fs.createWriteStream("input.txt");
output.write(O.length + "\n");
output.write(N.length + "\n");
for (var i=0; i<O.length; i++) {
    output.write("\"")
    for (var j=0; j<O[i].length; j++) {
        output.write(O[i][j][0] + "");
        if (j < O[i].length-1) {
            output.write(",");
        }
    }
    output.write("\"");
    if (i < O.length-1) {
        output.write(",")
    }
}
output.write("\n");
for (var i=0; i<N.length; i++) {
    output.write("\"")
    for (var j=0; j<N[i].length; j++) {
        output.write(N[i][j][0] + "");
        if (j < N[i].length-1) {
            output.write(",");
        }
    }
    output.write("\"");
    if (i < N.length-1) {
        output.write(",")
    }
}
output.end();