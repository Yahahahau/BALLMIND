/// <reference types="vite/client" />
declare module '*?worker&inline' {
  const WorkerCtor: new () => Worker;
  export default WorkerCtor;
}
