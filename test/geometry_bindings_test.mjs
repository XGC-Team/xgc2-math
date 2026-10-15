import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const {encodeGeometry,decodeGeometry}=await import(pathToFileURL(process.argv[3]));
const corpus=JSON.parse(execFileSync(process.argv[2],{encoding:'utf8'}));
for(const entry of corpus){const wire=encodeGeometry(entry.geometry);assert.equal(wire.type,entry.type);assert.deepEqual([...wire.parameters],entry.parameters);assert.deepEqual(decodeGeometry(wire),entry.geometry);
  for(const bad of [wire.parameters.slice(0,-1),Uint8Array.from([...wire.parameters,0])])assert.throws(()=>decodeGeometry({type:wire.type,parameters:bad}));
}
console.log(`TypeScript/C++: ${corpus.length} layouts byte-identical; flat triangles and malformed lengths pass`);
