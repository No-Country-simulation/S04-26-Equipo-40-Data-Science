/**
 * DataService.js — ConversaAI Dashboard
 *
 * PATTERN: Singleton
 * Guarantees one shared data source across all modules.
 * Emits 'data:filtered' events via EventBus when filters change.
 *
 * SOLID — Single Responsibility:
 * Manages ONLY data access, transformation, and seeding.
 * Does NOT touch the DOM or render anything.
 *
 * SOLID — Dependency Inversion:
 * Consumers depend on DataService (abstraction), not raw data arrays.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};

    /* ----------------------------------------------------------------
       Private helpers
       ---------------------------------------------------------------- */

    /** Generate a date series for the last N days */
    function _daysAgo(n) {
        const dates = [];
        for (let i = n; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            dates.push(d.toLocaleDateString('es-ES', { month: 'short', day: 'numeric' }));
        }
        return dates;
    }

    /** Seeded pseudo-random (deterministic) */
    function _seeded(seed, min, max) {
        const x = Math.sin(seed + 1) * 10000;
        return Math.floor((x - Math.floor(x)) * (max - min + 1)) + min;
    }

    /* ----------------------------------------------------------------
       Static dataset definitions
       ---------------------------------------------------------------- */

    const DATES = _daysAgo(29);

    /** Timeline: daily message counts per language */
    function _buildTimeline() {
        return DATES.map((date, i) => ({
            date,
            es: _seeded(i, 80, 220),
            pt: _seeded(i + 100, 40, 140)
        }));
    }

    /** Frustration timeline: average frustration score per day */
    function _buildFrustrationTimeline() {
        return DATES.map((date, i) => ({
            date,
            es: +(_seeded(i + 10, 18, 55) / 100).toFixed(2),
            pt: +(_seeded(i + 200, 20, 60) / 100).toFixed(2)
        }));
    }

    const SENTIMENT_DIST = {
        all: { positive: 58, neutral: 23, negative: 19 },
        es:  { positive: 61, neutral: 21, negative: 18 },
        pt:  { positive: 55, neutral: 25, negative: 20 }
    };

    const INTENT_DIST = [
        { label: 'Consulta',         es: 35, pt: 38 },
        { label: 'Reclamo',          es: 25, pt: 22 },
        { label: 'Facturación',      es: 18, pt: 16 },
        { label: 'Prob. Técnico',    es: 12, pt: 14 },
        { label: 'Cancelación',      es:  7, pt:  7 },
        { label: 'Resuelto',         es:  3, pt:  3 }
    ];

    const CHURN_DIST = {
        labels: ['Bajo', 'Medio', 'Alto', 'Crítico'],
        values: [58, 17, 17, 8]
    };

    const FRUSTRATION_BY_INTENT = {
        labels:  ['Cancelación', 'Reclamo', 'Prob. Técnico', 'Facturación', 'Consulta', 'Resuelto'],
        scores:  [0.82, 0.65, 0.54, 0.38, 0.22, 0.05]
    };

    const RECOMMENDATIONS = [
        {
            priority:  'critical',
            icon:      'fa-triangle-exclamation',
            title:     'Acción inmediata: retención proactiva',
            action:    '8% de conversaciones en riesgo crítico detectadas. Contactar por canal prioritario y ofrecer compensación o escalamiento a nivel superior.'
        },
        {
            priority:  'high',
            icon:      'fa-fire',
            title:     'Revisar calidad de atención — Alto riesgo de churn',
            action:    '17% de conversaciones con frustración alta. Asignar agentes senior y revisar scripts de respuesta para intenciones de reclamo y cancelación.'
        },
        {
            priority:  'medium',
            icon:      'fa-wrench',
            title:     'Problemas técnicos recurrentes',
            action:    'Escalado frecuente de "problema_tecnico". Crear base de conocimientos interna y mejorar el flujo de resolución automatizada.'
        },
        {
            priority:  'medium',
            icon:      'fa-file-invoice-dollar',
            title:     'Automatizar respuestas de facturación',
            action:    '18% de mensajes son consultas de facturación. Implementar chatbot de autoservicio para reducir carga del agente en un ~40%.'
        },
        {
            priority:  'low',
            icon:      'fa-chart-line',
            title:     'Monitorear métricas de XLM-R en producción',
            action:    'XLM-R está en fase de pruebas con 97.2% de precisión. Planificar despliegue gradual con A/B testing frente a RoBERTuito.'
        },
        {
            priority:  'low',
            icon:      'fa-language',
            title:     'Expandir cobertura en Portugués',
            action:    'BERTimbau cubre 73K reviews PT vs 200K de RoBERTuito en ES. Recolectar más datos PT para balancear entrenamiento.'
        }
    ];

    /* ----------------------------------------------------------------
       Singleton Implementation
       ---------------------------------------------------------------- */

    let _instance = null;

    function _createInstance() {
        let _currentLang = 'all';
        const _timeline   = _buildTimeline();
        const _frustTimeline = _buildFrustrationTimeline();

        return {
            /** Current language filter */
            getLang: () => _currentLang,

            /** Update the language filter and notify subscribers */
            setLang(lang) {
                _currentLang = lang;
                ConversaAI.EventBus.emit('data:filtered', { lang });
            },

            /** Get sentiment distribution for current or given lang */
            getSentimentDist(lang) {
                const l = lang || _currentLang;
                return SENTIMENT_DIST[l] || SENTIMENT_DIST.all;
            },

            /** Get intent distribution. Returns combined or per-lang values. */
            getIntentDist(lang) {
                const l = lang || _currentLang;
                return INTENT_DIST.map(item => ({
                    label: item.label,
                    value: l === 'all'
                        ? Math.round((item.es + item.pt) / 2)
                        : item[l] ?? Math.round((item.es + item.pt) / 2)
                }));
            },

            /** Get timeline data, optionally filtered by lang */
            getTimeline(lang) {
                const l = lang || _currentLang;
                return _timeline.map(row => ({
                    date: row.date,
                    es:   l !== 'pt' ? row.es : 0,
                    pt:   l !== 'es' ? row.pt : 0
                }));
            },

            /** Churn risk distribution (not filtered by lang) */
            getChurnDist: () => CHURN_DIST,

            /** Frustration by intent (not filtered by lang) */
            getFrustrationByIntent: () => FRUSTRATION_BY_INTENT,

            /** Frustration timeline */
            getFrustrationTimeline(lang) {
                const l = lang || _currentLang;
                return _frustTimeline.map(row => ({
                    date: row.date,
                    es:   l !== 'pt' ? row.es : null,
                    pt:   l !== 'es' ? row.pt : null
                }));
            },

            /** All recommendations */
            getRecommendations: () => RECOMMENDATIONS,

            /** All models (delegates to ModelRegistry) */
            getModels: () => ConversaAI.ModelRegistry.getAll(),

            /** Models filtered by current lang */
            getFilteredModels() {
                return ConversaAI.ModelRegistry.getByLang(_currentLang);
            }
        };
    }

    const DataService = {
        getInstance() {
            if (!_instance) _instance = _createInstance();
            return _instance;
        }
    };

    ConversaAI.DataService = DataService;

})(window);
