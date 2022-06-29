// import * as tf from "@tensorflow/tfjs";
window.addEventListener("load", () => {
    let canvas1 = document.getElementById('canvas1');
    const canvas2 = document.getElementById('canvas2');
    let ctx1 = canvas1.getContext("2d");
    const ctx2 = canvas2.getContext("2d");
    canvas1.height = 300;
    canvas1.width = 300;
    canvas2.height = 300;
    canvas2.width = 300;
    function getMousePos(canvas, evt) {
        var rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }

    let painting = false;
    function startPainting(e) {
        painting = true;
        draw(e);
    }

    function finishPainting() {
        painting = false;
        ctx1.beginPath();
    }

    ctx1.lineWidth = 25;
    ctx1.lineCap = "round";
    ctx1.strokeStyle = "black";
    function draw(e) {
        if (!painting) return;
        let pos = getMousePos(canvas1, e);
        ctx1.lineTo(pos.x, pos.y);
        ctx1.stroke();
    }


    //canvas event Listeners
    canvas1.addEventListener("mousedown", startPainting);
    canvas1.addEventListener("mouseup", finishPainting);
    canvas1.addEventListener("mousemove", draw);


    //button event listners
    const clr = document.getElementById('clr');
    const recog = document.getElementById('recog');
    //clear 
    function clear_canvas(canvas){
        let ctx = canvas.getContext('2d')
        ctx.fillStyle = "#FFFFFF"
        ctx.fillRect(0, 0, 300, 300)
    }
    clear_canvas(canvas1);
    clr.addEventListener("click",()=>{
        clear_canvas(canvas1);
        ctx2.clearRect(0,0,canvas2.width,canvas2.height);
    })
    recog.addEventListener("click",()=>{
        // var url =  canvas1.toDataURL("image/jpg");
        // console.log(url);
        ctx1 = canvas1.getContext("2d");
        var imageData = ctx1.getImageData(0,0,canvas1.width,canvas1.height);
        console.log(imageData)
        doPrediction(imageData);
    })
    

    // load model
    function loadModel(){
        tf.loadLayersModel("./model.json").then((model)=>{
            window.model = model;
        }); 
    }
    loadModel();
    //update canvas2 with pred
    function updatePred(pred){
        pred = pred.toString()
        ctx2.font = "200px Arial";
        ctx2.textAlign = "center";
        ctx2.fillText(pred,canvas2.width/2, 3*canvas2.height/4);
    }
    //do prediction
    function doPrediction(imageData){
        let input = tf.tidy(()=>{ //tidy helps in removing extra tensors
            let tensor =tf.browser.fromPixels(imageData,1)
            tensor = tensor.toFloat()
            tensor = tensor.div(tf.scalar(-255)); // to normalise
            tensor = tensor.add(tf.scalar(1)); // to invert
            tensor = tf.image.resizeBilinear(tensor,[28, 28]).mean(2).expandDims(-1).expandDims();
            return tensor;
        });
        tf.engine().startScope(); // work same as tidy
        // let x = window.model.predict([input]);input.print();let preds = x.dataSync().map((num)=>{ return (num*100).toPrecision(4);}); console.log("OUT: ", preds)

        window.model.predict([input]).array().then((scores)=>{
               scores = scores[0];
               var predicted = scores.indexOf(Math.max(...scores));
               updatePred(predicted);
        });

        tf.engine().endScope();
        input.dispose();
    }
});




