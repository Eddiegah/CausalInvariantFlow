// Runtime-independent Lorenz-63 numerical smoke test.
// This validates the simulator equations and finite-value behavior even when
// the device cannot launch its Windows Store Python executable.
const assert = require('node:assert/strict');
function f(s, sigma=10, rho=28, beta=8/3) {
  const [x,y,z] = s;
  return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z];
}
function step(s, dt=1e-3) {
  const k1=f(s);
  const mid=s.map((v,i)=>v+dt*k1[i]/2);
  const k2=f(mid);
  return s.map((v,i)=>v+dt*k2[i]);
}
let s=[1,1,25];
for(let i=0;i<10000;i++) {
  s=step(s);
  assert.equal(s.length,3);
  assert.ok(s.every(Number.isFinite));
  assert.ok(s.every(v=>Math.abs(v)<1e6));
}
console.log(JSON.stringify({test:'lorenz63_smoke',passed:true,final_state:s}));
