import { Component, ViewChild, OnInit } from '@angular/core';

import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent implements OnInit {
  
  linearModel: tf.Sequential;
  prediction: any;
  title = 'app';


  ngOnInit() {
    this.trainNewModel();
  }

  async trainNewModel() {
    //Define new linear model for regression
    this.linearModel = tf.sequential();
    //Add a new layer to empty model with 1 input, 1 output
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));

    //Prepare the new model for training by specifying loss and optimizer.
    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    //Model is finished, now train it!
    //First, fill it with random data...
    const xs = tf.tensor1d([3.2, 4.4, 5.5, 2.2, 4.8, 1.9, 6.4, 6.1, 4.5]);
    const ys = tf.tensor1d([1.6, 2.7, 3.5, 9.9, 1.2, 4.2, 4.5, 9.8, 7.5]);

    //Then train
    await this.linearModel.fit(xs, ys);

    console.log('model training!')

  }

  linearPrediction(val) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1, 1])) as any;
    this.prediction = Array.from(output.dataSync())[0]



  }

}
