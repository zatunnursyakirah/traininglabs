//package path
package ai.certifai.training.feedforward;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.weightinit.WeightInit;

import java.io.IOException;

import static org.nd4j.weightinit.WeightInit.*;

public class MnistFNN {

    final static int seed = 1234;
    final static int batchSize = 500;
    final static int epoch = 1;

    public static void main(String[] args) throws IOException {
        //mnistdatasetiterator
        MnistDataSetIterator trainMnist = new MnistDataSetIterator(batchSize, true, seed);
        MnistDataSetIterator testMnist = new MnistDataSetIterator(batchSize, false, seed);

        //do normalize
        //scaler minmax
        //technique 1
//        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
//        DataSet trainSet = trainMnist.next();
//        scaler.fit(trainSet);
//        scaler.transform(trainSet);

        //normalization or this technique
        //technique 2
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(trainMnist);
        trainMnist.setPreProcessor(scaler);
        testMnist.setPreProcessor(scaler);

//        System.out.println(trainMnist.next().numExamples());
//        System.out.println(((DataSet) trainSet).numExamples());

        //model config
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3))
                .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainMnist.inputColumns())
                        .nOut(124)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(282)
                        .build())
                .layer(new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(trainMnist.totalOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(1000));

        for (int i = 0; i <= epoch; i++) {
            model.fit(trainMnist);
        }

        Evaluation evalTrain = model.evaluate(trainMnist);
        Evaluation evalTest = model.evaluate(testMnist);

        System.out.println(evalTrain.stats());
        System.out.println(evalTest.stats());

    }


}
