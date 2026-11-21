// Passive display tracers integrated through imported lattice velocities.
class VelocityTracers {
  constructor(THREE,shape,spacing){
    this.THREE=THREE;this.shape=shape;this.spacing=spacing;
    this.count=900;this.tail=18;this.randomState=42;this.head=0;this.travel=0;
    this.positions=new Float32Array(this.count*3);
    this.history=new Float32Array(this.count*this.tail*3);
    this.lines=new Float32Array(this.count*(this.tail-1)*6);
    const colors=new Float32Array(this.lines.length);
    for(let i=0;i<this.count;i++)for(let j=0;j<this.tail-1;j++)for(let end=0;end<2;end++){
      const offset=(i*(this.tail-1)+j)*6+end*3;
      const fade=Math.pow(1-(j+end)/this.tail,1.8);
      colors[offset]=.12*fade;colors[offset+1]=.80*fade;colors[offset+2]=fade;
    }
    this.lineGeometry=new THREE.BufferGeometry();
    this.lineGeometry.setAttribute('position',new THREE.BufferAttribute(this.lines,3).setUsage(THREE.DynamicDrawUsage));
    this.lineGeometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
    this.group=new THREE.Group();
    this.group.add(new THREE.LineSegments(this.lineGeometry,new THREE.LineBasicMaterial({
      vertexColors:true,transparent:true,opacity:.9,blending:THREE.AdditiveBlending,depthWrite:false
    })));
    this.headGeometry=new THREE.BufferGeometry();
    this.headGeometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(this.count*3),3).setUsage(THREE.DynamicDrawUsage));
    this.group.add(new THREE.Points(this.headGeometry,new THREE.PointsMaterial({
      color:0xa8f6ff,size:2.5,sizeAttenuation:false,transparent:true,opacity:.9,depthWrite:false
    })));
    this.group.children.forEach(object=>object.frustumCulled=false);
  }
  random(){
    this.randomState=(1664525*this.randomState+1013904223)>>>0;
    return this.randomState/4294967296;
  }
  fluid(x,y,z,mask){
    const [nx,ny,nz]=this.shape;
    if(y<.5||y>ny-1.5||z<.5||z>nz-1.5)return false;
    const ix=((Math.round(x)%nx)+nx)%nx;
    return !mask||mask[ix+nx*(Math.round(y)+ny*Math.round(z))]===0;
  }
  seed(i,mask,upstream){
    const [nx,ny,nz]=this.shape,offset=3*i;
    let x,y,z,valid=false;
    for(let attempt=0;attempt<100;attempt++){
      x=upstream ? .5 : this.random()*(nx-1);
      y=(ny-1)*(.18+.64*this.random());z=(nz-1)*(.18+.64*this.random());
      if(this.fluid(x,y,z,mask)){valid=true;break;}
    }
    // A narrow channel may require a broader search.
    for(let c=0;!valid&&c<nx*ny*nz;c++)if(!mask||mask[c]===0){
      x=c%nx;y=Math.floor(c/nx)%ny;z=Math.floor(c/(nx*ny));valid=this.fluid(x,y,z,mask);
    }
    this.positions.set([x,y,z],offset);
    for(let j=0;j<this.tail;j++)this.history.set([x,y,z],(i*this.tail+j)*3);
  }
  reset(mask){
    this.head=0;this.travel=0;
    for(let i=0;i<this.count;i++)this.seed(i,mask,false);
    this.draw();
  }
  velocity(x,y,z,sample,mask){
    const [nx,ny,nz]=this.shape;
    const ix=Math.floor(x),iy=Math.floor(y),iz=Math.floor(z),a=x-ix,b=y-iy,c=z-iz;
    let ux=0,uy=0,uz=0;
    for(let dz=0;dz<2;dz++)for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){
      const xx=((ix+dx)%nx+nx)%nx,yy=Math.max(0,Math.min(ny-1,iy+dy)),zz=Math.max(0,Math.min(nz-1,iz+dz));
      const index=xx+nx*(yy+ny*zz);
      if(mask&&mask[index]!==0)continue;
      const weight=(dx?a:1-a)*(dy?b:1-b)*(dz?c:1-c);
      ux+=weight*sample('velocity',3*index);
      uy+=weight*sample('velocity',3*index+1);
      uz+=weight*sample('velocity',3*index+2);
    }
    return [ux,uy,uz];
  }
  advance(dt,sample,mask,maxSpeed){
    if(dt<=0)return;
    const nx=this.shape[0];
    const substeps=Math.max(1,Math.ceil(maxSpeed*dt/.2)),h=dt/substeps;
    for(let i=0;i<this.count;i++){
      const offset=3*i;
      let x=this.positions[offset],y=this.positions[offset+1],z=this.positions[offset+2],reseed=false;
      for(let sub=0;sub<substeps;sub++){
        const u=this.velocity(x,y,z,sample,mask);
        const v=this.velocity(x+.5*h*u[0],y+.5*h*u[1],z+.5*h*u[2],sample,mask);
        const xx=x+h*v[0],yy=y+h*v[1],zz=z+h*v[2];
        if(xx<-.5||xx>=nx-.5||!this.fluid(xx,yy,zz,mask)){reseed=true;break;}
        this.travel+=Math.hypot(xx-x,yy-y,zz-z);
        x=xx;y=yy;z=zz;
      }
      if(reseed)this.seed(i,mask,true);
      else this.positions.set([x,y,z],offset);
    }
    this.head=(this.head+1)%this.tail;
    for(let i=0;i<this.count;i++)this.history.set(this.positions.subarray(3*i,3*i+3),(i*this.tail+this.head)*3);
    this.draw();
  }
  world(source,index,target,offset){
    const [nx,ny,nz]=this.shape,[dx,dy,dz]=this.spacing;
    target[offset]=(source[index]-(nx-1)/2)*dx;
    target[offset+1]=(source[index+2]-(nz-1)/2)*dz;
    target[offset+2]=(source[index+1]-(ny-1)/2)*dy;
  }
  draw(){
    const heads=this.headGeometry.attributes.position.array;
    for(let i=0;i<this.count;i++){
      this.world(this.positions,3*i,heads,3*i);
      for(let j=0;j<this.tail-1;j++){
        const current=(this.head-j+this.tail)%this.tail,next=(current-1+this.tail)%this.tail;
        const offset=(i*(this.tail-1)+j)*6;
        this.world(this.history,(i*this.tail+current)*3,this.lines,offset);
        this.world(this.history,(i*this.tail+next)*3,this.lines,offset+3);
      }
    }
    this.lineGeometry.attributes.position.needsUpdate=true;
    this.headGeometry.attributes.position.needsUpdate=true;
  }
}
