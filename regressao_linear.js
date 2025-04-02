const tf = require('@tensorflow/tfjs-node');
const Papa = require('papaparse');
const fs = require('fs');

// 📂 Lendo arquivo CSV
const csvFilePath = 'ocorrencias_registradas.csv';
const file = fs.readFileSync(csvFilePath, 'utf8');

let data = [];
Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    complete: (result) => {
        data = result.data;

        console.log("Cabeçalhos do CSV:", Object.keys(data[0] || {}));

        // 🔹 Filtrar apenas os dados numéricos válidos
        data = data.filter(row =>
            !isNaN(row.furto_outros) &&
            !isNaN(row.furto_de_veiculo) &&
            row.furto_outros !== null &&
            row.furto_de_veiculo !== null
        );

        const xValues = data.map(row => row.furto_outros);
        const yValues = data.map(row => row.furto_de_veiculo);

        console.log('Dados processados:', xValues.slice(0, 10), yValues.slice(0, 10)); // Exibir amostra

        if (xValues.length === 0 || yValues.length === 0) {
            console.error("Erro: Nenhum dado válido encontrado.");
            return;
        }

        // 🔹 Normalização (Evita problemas numéricos no TensorFlow)
        const xMin = xValues.reduce((min, val) => val < min ? val : min, Infinity);
        const xMax = xValues.reduce((max, val) => val > max ? val : max, -Infinity);
        const yMin = yValues.reduce((min, val) => val < min ? val : min, Infinity);
        const yMax = yValues.reduce((max, val) => val > max ? val : max, -Infinity);

        const xNormalized = xValues.map(x => (x - xMin) / (xMax - xMin));
        const yNormalized = yValues.map(y => (y - yMin) / (yMax - yMin));

        // 🔹 Criar tensores
        tf.util.shuffle(data);
        const xs = tf.tensor2d(xNormalized.map(v => [v]), [xNormalized.length, 1]);
        const ys = tf.tensor2d(yNormalized.map(v => [v]), [yNormalized.length, 1]);

        // 🔹 Criar modelo
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
        model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

        // 🔹 Treinar modelo
        async function trainModel() {
            console.log('Treinando modelo...');
            await model.fit(xs, ys, { epochs: 10 }); // Mais épocas para melhor ajuste
            console.log('Treinamento concluído');

            // 🔹 Fazer uma previsão
            const testValue = (10 - xMin) / (xMax - xMin); // Normalizar entrada
            const prediction = model.predict(tf.tensor2d([[testValue]], [1, 1]));
            prediction.print();

            // 🔹 Coletar pesos da equação da reta y = ax + b
            const weights = model.getWeights();
            const aTensor = await weights[0].array(); // Inclinação (a)
            const bTensor = await weights[1].array(); // Intercepto (b)

            const a = aTensor[0][0] * (yMax - yMin) / (xMax - xMin);
            const b = bTensor[0] * (yMax - yMin) + yMin - a * xMin;

            console.log(`Equação da reta: y = ${a.toFixed(4)}x + ${b.toFixed(4)}`);

            // 🔹 Calcular R²
            const yPredicted = xValues.map(x => a * x + b);
            const ssTotal = yValues.reduce((sum, y) => sum + Math.pow(y - (yValues.reduce((a, b) => a + b) / yValues.length), 2), 0);
            const ssResidual = yValues.reduce((sum, y, i) => sum + Math.pow(y - yPredicted[i], 2), 0);
            const r2 = 1 - (ssResidual / ssTotal);

            console.log(`Coeficiente de determinação (R²): ${r2.toFixed(4)}`);
        }

        trainModel();
    }
});
