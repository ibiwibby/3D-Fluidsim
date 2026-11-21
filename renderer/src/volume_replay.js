// Surface nets of the 50% liquid-fraction isosurface from a CPU or GPU solve.
class VolumeReplay {
  static async load(data,material) {
    async function decode(s,Type){
      const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));
      const raw=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
      return new Type(await new Response(raw).arrayBuffer());
    }
    const [cubes,samples,heights,ducks]=await Promise.all([
      decode(data.cubes,Int32Array),decode(data.samples,Uint8Array),
      decode(data.heights,Int16Array),decode(data.ducks,Int16Array)]);
    const replay=new VolumeReplay(data,cubes,samples,heights,ducks,material);
    if(data.wake)[replay.wake,replay.heading]=await Promise.all([decode(data.wake,Uint8Array),decode(data.heading,Int16Array)]);
    return replay;
  }
  constructor(data,cubes,samples,heights,ducks,material){
    this.meta=data.meta;this.steps=data.steps;this.samples=samples;this.heights=heights;this.ducks=ducks;
    const {nx,ny,nz}=this.meta,sz=nx*ny,len=cubes.length;
    const offsets=[0,1,nx,nx+1,sz,sz+1,sz+nx,sz+nx+1];
    const pointMap=new Int32Array(sz*nz),cubeMap=new Int32Array(sz*nz).fill(-1);
    this.corner=new Int32Array(len*8);this.base=new Float32Array(len*3);
    this.neighbors=new Int32Array(len*9).fill(-1);
    for(let i=0;i<len;i++){
      const c=cubes[i];cubeMap[c]=i;
      this.base[i*3]=c%nx-(nx-1)/2;this.base[i*3+1]=Math.floor(c/sz);
      this.base[i*3+2]=Math.floor(c/nx)%ny-(ny-1)/2;
      for(const d of offsets)pointMap[c+d]=1;
    }
    let n=0;for(let i=0;i<pointMap.length;i++)if(pointMap[i])pointMap[i]=n++;
    if(samples.length!==n*this.steps.length)throw Error('Invalid volume samples');
    const moving=this.meta.worldScale?new Uint8Array(n):null;
    for(let j=n;j<samples.length;j++){if(moving&&samples[j])moving[j%n]=1;samples[j]+=samples[j-n];}
    this.field=new Float32Array(n);
    for(let i=0;i<len;i++){
      const c=cubes[i];
      offsets.forEach((off,j)=>this.corner[i*8+j]=pointMap[c+off]);
      for(const [axis,a,b] of [[0,nx,sz],[1,sz,1],[2,1,nx]]){
        const k=i*9+axis*3;
        this.neighbors[k]=cubeMap[c-a]??-1;this.neighbors[k+1]=cubeMap[c-a-b]??-1;this.neighbors[k+2]=cubeMap[c-b]??-1;
      }
    }
    this.firstMoving=0;this.staticIndices=0;
    if(moving){
      this.order=new Uint32Array(len);
      let front=0,back=len;
      for(let i=0;i<len;i++){
        let dynamic=false;
        for(let j=0;j<8&&!dynamic;j++)dynamic=Boolean(moving[this.corner[8*i+j]]);
        if(dynamic)this.order[--back]=i;else this.order[front++]=i;
      }
      this.firstMoving=front;
    }
    this.pos=new THREE.BufferAttribute(new Float32Array(len*3),3).setUsage(THREE.DynamicDrawUsage);
    this.normal=this.pos.clone().setUsage(THREE.DynamicDrawUsage);
    this.cross=new Uint8Array(len);
    this.indices=new THREE.BufferAttribute(new Uint32Array(len*18),1).setUsage(THREE.DynamicDrawUsage);
    this.g=new THREE.BufferGeometry();this.g.setAttribute('position',this.pos);this.g.setAttribute('normal',this.normal);
    this.g.setIndex(this.indices);this.g.setDrawRange(0,0);
    this.mesh=new THREE.Mesh(this.g,material);this.mesh.frustumCulled=false;this.key=-1;
    this.mesh.scale.setScalar(this.meta.worldScale||1);
  }
  update(k,fraction){
    // Quantizing time limits rebuilds when playback is much slower than refresh.
    const part=Math.round(fraction*4),key=k*4+part;if(key===this.key)return;
    const first=this.key<0?0:this.firstMoving;
    this.key=key;
    const a=part/4,n=this.field.length,f=this.field,s=this.samples;
    const j=k*n,l=Math.min(k+1,this.steps.length-1)*n;
    for(let i=0;i<n;i++)f[i]=s[j+i]+a*(s[l+i]-s[j+i]);
    const values=new Float32Array(8),edges=[0,1,2,3,4,5,6,7,0,2,1,3,4,6,5,7,0,4,1,5,2,6,3,7];
    for(let slot=first;slot<this.cross.length;slot++){
      const i=this.order?this.order[slot]:slot;
      let mask=0;
      for(let p=0;p<8;p++){values[p]=f[this.corner[8*i+p]];if(values[p]>=127.5)mask|=1<<p;}
      this.cross[i]=mask;if(mask===0||mask===255)continue;
      let x=0,y=0,z=0,count=0;
      for(let e=0;e<24;e+=2){
        const p=edges[e],q=edges[e+1];if(((mask>>p)^(mask>>q))%2===0)continue;
        const t=(127.5-values[p])/(values[q]-values[p]);
        x+=(p&1)+((q&1)-(p&1))*t;y+=((p>>1)&1)+(((q>>1)&1)-((p>>1)&1))*t;
        z+=(p>>2)+((q>>2)-(p>>2))*t;count++;
      }
      this.pos.setXYZ(i,this.base[3*i]+x/count,this.base[3*i+1]+z/count,this.base[3*i+2]+y/count);
      const [v0,v1,v2,v3,v4,v5,v6,v7]=values;
      const gx=v0+v2+v4+v6-v1-v3-v5-v7,gy=v0+v1+v4+v5-v2-v3-v6-v7,gz=v0+v1+v2+v3-v4-v5-v6-v7;
      const mag=Math.hypot(gx,gy,gz)||1;this.normal.setXYZ(i,gx/mag,gz/mag,gy/mag);
    }
    let write=first?this.staticIndices:0;const out=this.indices.array;
    for(let slot=first;slot<this.cross.length;slot++){
      if(slot===this.firstMoving)this.staticIndices=write;
      const i=this.order?this.order[slot]:slot;
      const mask=this.cross[i];if(!mask||mask===255)continue;
      for(let axis=0;axis<3;axis++){
        if(((mask^(mask>>(1<<axis)))&1)===0)continue;
        const at=i*9+axis*3,mid=this.neighbors[at+1];
        let left=this.neighbors[at],right=this.neighbors[at+2];
        if(left<0||mid<0||right<0)continue;
        if(mask&1){const tmp=left;left=right;right=tmp;}
        out[write++]=i;out[write++]=left;out[write++]=mid;
        out[write++]=i;out[write++]=mid;out[write++]=right;
      }
    }
    if(this.firstMoving===this.cross.length)this.staticIndices=write;
    this.pos.needsUpdate=this.normal.needsUpdate=this.indices.needsUpdate=true;this.g.setDrawRange(0,write);
  }
}

function volumeFittings(m,nickel,brass,dark){
  const root=new THREE.Group(),ox=(m.nx-1)/2,oz=(m.ny-1)/2;
  function add(g,mat,x,z,y){
    const piece=new THREE.Mesh(g,mat);piece.position.set(x-ox,z,y-oz);root.add(piece);
    piece.castShadow=piece.receiveShadow=true;return piece;
  }
  const [tx,ty,tz,tr]=m.ports[0],curve=new THREE.CurvePath(),top=tz+tr+5;
  const p=(x,z)=>new THREE.Vector3(x,z,0);
  curve.add(new THREE.LineCurve3(p(-5,-3),p(-5,top-4)));
  curve.add(new THREE.QuadraticBezierCurve3(p(-5,top-4),p(-5,top),p(-1,top)));
  curve.add(new THREE.LineCurve3(p(-1,top),p(tx-5,top)));
  curve.add(new THREE.QuadraticBezierCurve3(p(tx-5,top),p(tx,top),p(tx,top-5)));
  curve.add(new THREE.LineCurve3(p(tx,top-5),p(tx,tz+2)));
  add(new THREE.TubeGeometry(curve,80,tr*.9,32,false),nickel,0,0,ty);
  add(new THREE.CylinderGeometry(tr+1.15,tr+1.15,2,48,1,true),brass,tx,tz+1,ty);
  const [hx,hy,hz]=m.hub,ring=Math.hypot(m.ports[1][0]-hx,m.ports[1][1]-hy);
  add(new THREE.CylinderGeometry(.025*m.nx,.025*m.nx,hz,32),nickel,hx,hz/2,hy);
  add(new THREE.CylinderGeometry(ring+m.ports[1][3]+1.2,ring+m.ports[1][3]+1.2,1,64),dark,hx,hz,hy);
  for(const [x,y,z,r] of m.ports.slice(1)){
    add(new THREE.CylinderGeometry(r+1.15,r+1.15,2.2,32,1,true),brass,x,z-.5,y);
    const lip=add(new THREE.RingGeometry(r,r+1.15,40),brass,x,z+.6,y);lip.rotation.x=-Math.PI/2;
  }
  return root;
}
