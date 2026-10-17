let running = false;
let iterations = 0;
let checksum = 0;

function work(): void {
  if (!running) return;
  let value = checksum || 0.6180339887498948;
  for (let index = 0; index < 180_000; index += 1) {
    value = Math.sin(value * 1.0000001192092896 + index * 0.000001) + 1.125;
  }
  checksum = value;
  iterations += 1;
  if (iterations % 4 === 0) self.postMessage({ iterations, checksum });
  setTimeout(work, 0);
}

self.onmessage = (event: MessageEvent<{ kind: "start" | "stop" }>) => {
  if (event.data.kind === "stop") {
    running = false;
    self.close();
    return;
  }
  if (running) return;
  running = true;
  work();
};

export {};
