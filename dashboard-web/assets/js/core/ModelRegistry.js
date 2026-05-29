/**
 * ModelRegistry.js — ConversaAI Dashboard
 *
 * PATTERN: Registry + Open/Closed Principle
 * The registry is CLOSED for modification but OPEN for extension:
 * New models can be added by calling ModelRegistry.register() without
 * touching existing model definitions or consuming code.
 *
 * SOLID — Single Responsibility:
 * This file ONLY manages the catalog of available models.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};

    /**
     * @typedef {Object} ModelDefinition
     * @property {string}   id          - Unique identifier
     * @property {string}   name        - Display name
     * @property {string}   repo        - HuggingFace repo or identifier
     * @property {string}   architecture- Model architecture
     * @property {string}   lang        - Language(s): "es" | "pt" | "es+pt"
     * @property {number}   accuracy    - Test accuracy (0–100)
     * @property {number}   precision   - Precision (0–100)
     * @property {number}   recall      - Recall (0–100)
     * @property {number}   f1Score     - F1 Score (0–100)
     * @property {number}   latencyMs   - Inference latency in milliseconds
     * @property {number}   trainingSamples - Number of training samples
     * @property {string}   status      - "active" | "testing" | "baseline"
     * @property {string}   useCase     - Primary use case description
     * @property {Object}   f1ByClass   - { positive, negative, neutral }
     */

    const ModelRegistry = (function () {
        /** @type {Map<string, ModelDefinition>} */
        const _registry = new Map();

        return {
            /**
             * Register a new model definition.
             * OCP: This is the only extension point; existing entries are immutable.
             * @param {ModelDefinition} model
             */
            register(model) {
                if (!model.id) throw new Error('[ModelRegistry] Model must have an id.');
                if (_registry.has(model.id)) {
                    console.warn(`[ModelRegistry] Overwriting model "${model.id}"`);
                }
                _registry.set(model.id, Object.freeze({ ...model }));
            },

            /**
             * Retrieve a model by id.
             * @param  {string} id
             * @returns {ModelDefinition|undefined}
             */
            get(id) {
                return _registry.get(id);
            },

            /**
             * Retrieve all registered models as an array.
             * @returns {ModelDefinition[]}
             */
            getAll() {
                return Array.from(_registry.values());
            },

            /**
             * Filter models by language.
             * @param {string} lang - "es" | "pt" | "all"
             * @returns {ModelDefinition[]}
             */
            getByLang(lang) {
                if (lang === 'all') return this.getAll();
                return this.getAll().filter(m =>
                    m.lang === lang || m.lang === 'es+pt'
                );
            }
        };
    })();

    /* ----------------------------------------------------------------
       Register the 4 models from the ConversaAI project
       OCP: Future models just need a ModelRegistry.register({...}) call
       ---------------------------------------------------------------- */

    ModelRegistry.register({
        id:               'tfidf-logreg',
        name:             'TF-IDF + LogReg',
        repo:             'scikit-learn',
        architecture:     'TF-IDF + Logistic Regression',
        lang:             'es+pt',
        accuracy:         82.5,
        precision:        81.2,
        recall:           80.8,
        f1Score:          81.0,
        latencyMs:        10,
        trainingSamples:  271000,
        status:           'baseline',
        useCase:          'Benchmark de comparativa',
        f1ByClass:        { positive: 84.5, negative: 79.2, neutral: 72.0 }
    });

    ModelRegistry.register({
        id:               'robertuito-es',
        name:             'RoBERTuito (ES)',
        repo:             'pysentimiento/robertuito-sentiment-analysis',
        architecture:     'RoBERTa Transformer',
        lang:             'es',
        accuracy:         96.1,
        precision:        95.8,
        recall:           96.3,
        f1Score:          96.0,
        latencyMs:        145,
        trainingSamples:  200000,
        status:           'active',
        useCase:          'Análisis de sentimiento en Español',
        f1ByClass:        { positive: 97.1, negative: 95.8, neutral: 93.2 }
    });

    ModelRegistry.register({
        id:               'bertimbau-pt',
        name:             'BERTimbau (PT)',
        repo:             'pysentimiento/bertimbau-sentiment',
        architecture:     'BERT Transformer',
        lang:             'pt',
        accuracy:         95.8,
        precision:        95.2,
        recall:           95.9,
        f1Score:          95.5,
        latencyMs:        152,
        trainingSamples:  73000,
        status:           'active',
        useCase:          'Análisis de sentimiento en Portugués',
        f1ByClass:        { positive: 96.4, negative: 95.1, neutral: 91.8 }
    });

    ModelRegistry.register({
        id:               'xlmr-multilang',
        name:             'XLM-R Fine-tuned',
        repo:             'Rosela/xlmr-intent-6clases',
        architecture:     'XLM-RoBERTa Transformer',
        lang:             'es+pt',
        accuracy:         97.2,
        precision:        97.0,
        recall:           97.3,
        f1Score:          97.1,
        latencyMs:        160,
        trainingSamples:  271000,
        status:           'testing',
        useCase:          'Sentimiento + Intención multilenguaje',
        f1ByClass:        { positive: 98.0, negative: 97.2, neutral: 95.0 }
    });

    ConversaAI.ModelRegistry = ModelRegistry;

})(window);
