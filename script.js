const GRID_SIZE = 32;
const UPDATE_INTERVAL = 250;
const WORKGROUP_SIZE = 8;

const canvas = document.querySelector("canvas");

// WebGPU device initialization
if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
}

const device = await adapter.requestDevice();

// Canvas configuration
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});

// Create a buffer with the vertices for a single cell.
const vertices = new Float32Array([
    -0.8, -0.8,
    0.8, -0.8,
    0.8, 0.8,

    -0.8, -0.8,
    0.8, 0.8,
    -0.8, 0.8,
]);
const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2",
        offset: 0,
        shaderLocation: 0, // Position. Matches @location(0) in the @vertex shader.
    }],
};

// Create the bind group layout and pipeline layout.
const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: {} // Grid uniform buffer
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } // Cell state input buffer
    }, {
        binding: 2,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } // Cell state input buffer
    },
    {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } // Cell state output buffer
    },
    {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } // Cell state output buffer
    }]
});

const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
});

// Create the shader that will render the cells.
const cellShaderModule = device.createShaderModule({
    label: "Cell shader",
    code: `
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) cell: vec2f,
            @location(1) colour: vec4f,
          };

          @group(0) @binding(0) var<uniform> grid: vec2f;
          @group(0) @binding(1) var<storage> cellState: array<u32>;
          @group(0) @binding(2) var<storage> activeNeighbors: array<u32>;

          @vertex
          fn vertexMain(@location(0) position: vec2f,
                        @builtin(instance_index) instance: u32) -> VertexOutput {
            var output: VertexOutput;

            let i = f32(instance);
            let cell = vec2f(i % grid.x, floor(i / grid.x));

            let scale = f32(cellState[instance]);
            let cellOffset = cell / grid * 2;
            let gridPos = (position*scale+1) / grid - 1 + cellOffset;

            output.position = vec4f(gridPos, 0, 1);
            output.cell = cell / grid;


            let temp = u32(activeNeighbors[instance]);





            if(temp == 1){
                output.colour = vec4f(0, 0, 1 , 1);
            }
            
            else if(temp == 2){
                output.colour = vec4f(0.772, 0.772, 0.929, 1);
                // output.colour = vec4f(0.2196, 0.2235, 0.2353, 1);
                output.colour = vec4f(1, 1, 1, 1);
            }
            
            else if(temp == 3){
                // output.colour = vec4f(0.710, 0.788, 0.604, 1);
                output.colour = vec4f(0.9843, 0.9843, 0.5294, 1);
            }
            
            else if(temp == 4){
                output.colour = vec4f(0.5, 0.5, 1 , 1);
            }
            
            else if(temp == 5){
                output.colour = vec4f(0, 0, 1, 1);
            } 
            else if(temp == 6){
                output.colour = vec4f(0, 0, 0, 1);
            } 
            else if(temp == 7){
            } 
            else if(temp == 8){
                output.colour = vec4f(1, 0, 0, 1);
            } 
            else{
               output.colour = vec4f(1, 1, 1, 1);
            }

            return output;
          }

          @fragment
          fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            // return vec4f(input.cell, 1.0 - input.cell.x, 1);
            // return vec4f(1, input.colour, 0, 1);
            // return vec4f(1, 0.5, 0, 1);
            return input.colour;
          }
        `
});

// Create a pipeline that renders the cell.
const cellPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: pipelineLayout,
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout]
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat
        }]
    }
});

// Create the compute shader that will process the game of life simulation.
const simulationShaderModule = device.createShaderModule({
    label: "Life simulation shader",
    code: `
          @group(0) @binding(0) var<uniform> grid: vec2f;

          @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
          @group(0) @binding(2) var<storage> activeNeighborsInput: array<u32>;
          @group(0) @binding(3) var<storage, read_write> cellStateOut: array<u32>;
          @group(0) @binding(4) var<storage, read_write> activeNeighborsOutput: array<u32>;

          fn cellIndex(cell: vec2u) -> u32 {
            return (cell.y % u32(grid.y)) * u32(grid.x) +
                   (cell.x % u32(grid.x));
          }

          fn cellActive(x: u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2(x, y))];
          }

          @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
          fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            // Determine how many active neighbors this cell has.
            let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                                  cellActive(cell.x+1, cell.y) +
                                  cellActive(cell.x+1, cell.y-1) +
                                  cellActive(cell.x, cell.y-1) +
                                  cellActive(cell.x-1, cell.y-1) +
                                  cellActive(cell.x-1, cell.y) +
                                  cellActive(cell.x-1, cell.y+1) +
                                  cellActive(cell.x, cell.y+1);

            let i = cellIndex(cell.xy);

            activeNeighborsOutput[i] = activeNeighbors;

            // Conway's game of life rules:
            switch activeNeighbors {
              case 2: { // Active cells with 2 neighbors stay active.
                cellStateOut[i] = cellStateIn[i];
              }
              case 3: { // Cells with 3 neighbors become or stay active.
                cellStateOut[i] = 1;
              }
            //   case 4: { // Cells with 3 neighbors become or stay active.
            //     cellStateOut[i] = 1;
            //   }
            //   case 5: { // Cells with 3 neighbors become or stay active.
            //     cellStateOut[i] = 1;
            //   }
            //   case 6: { // Cells with 3 neighbors become or stay active.
            //     cellStateOut[i] = 1;
            //   }
            //   case 7: { // Cells with 3 neighbors become or stay active.
            //     cellStateOut[i] = 1;
            //   }
              
              
              default: { // Cells with < 2 or > 3 neighbors become inactive.
                cellStateOut[i] = 0;
              }
            }
          }
        `
});

// Create a compute pipeline that updates the game state.
const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
    }
});

// Create a uniform buffer that describes the grid.
const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

// Create an array representing the active state of each cell.
const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);


const activeNeighborsBuffer = [
    device.createBuffer({
        label: "active buffer State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
        label: "active buffer State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
];









// Create two storage buffers to hold the cell state.
const cellStateStorage = [
    device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
];

// Set each cell to a random state, then copy the JavaScript array into
// the storage buffer.
for (let i = 0; i < cellStateArray.length; ++i) {
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

// Create a bind group to pass the grid uniforms into the pipeline
const bindGroups = [
    device.createBindGroup({
        label: "Cell renderer bind group A",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }, {
            binding: 1,
            resource: { buffer: cellStateStorage[0] }
        }, {
            binding: 2,
            resource: { buffer: activeNeighborsBuffer[0] }
        }, {
            binding: 3,
            resource: { buffer: cellStateStorage[1] }
        }, {
            binding: 4,
            resource: { buffer: activeNeighborsBuffer[1] }
        }],
    }),
    device.createBindGroup({
        label: "Cell renderer bind group B",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }, {
            binding: 1,
            resource: { buffer: cellStateStorage[1] }
        }, {
            binding: 2,
            resource: { buffer: activeNeighborsBuffer[1] }
        }, {
            binding: 3,
            resource: { buffer: cellStateStorage[0] }
        }, {
            binding: 4,
            resource: { buffer: activeNeighborsBuffer[0] }
        }],
    }),
];

let step = 0;
async function updateGrid() {
    const encoder = device.createCommandEncoder();

    // Start a compute pass
    const computePass = encoder.beginComputePass();

    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);
    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
    computePass.end();

    step++; // Increment the step count

    // Start a render pass
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
            // clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
            // clearValue: { r: 0.710, g: 0.788, b: 0.604, a: 1.0 },
            storeOp: "store",
        }]
    });

    // Draw the grid.
    pass.setPipeline(cellPipeline);
    pass.setBindGroup(0, bindGroups[step % 2]); // Updated!
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);

    // End the render pass and submit the command buffer
    pass.end();


    device.queue.submit([encoder.finish()]);


}
// setInterval(updateGrid, UPDATE_INTERVAL);

updateGrid();


const playPauseButton = document.getElementById('playPauseButton');
// const prevButton = document.getElementById('prevButton');
const nextButton = document.getElementById('nextButton');
const element = document.getElementById("play_push");

// Add event listeners for play/pause functionality
var intervalID;
var flag = true;
playPauseButton.addEventListener('click', () => {


    if (flag) {

        intervalID = setInterval(updateGrid, UPDATE_INTERVAL);
        element.innerHTML = "push";
        flag = false;
    } else {
        clearInterval(intervalID);
        element.innerHTML = "play";
        flag = true;
    }
    updateGrid();
});

// prevButton.addEventListener('click', () => {

// });

nextButton.addEventListener('click', () => {
    updateGrid();
});
