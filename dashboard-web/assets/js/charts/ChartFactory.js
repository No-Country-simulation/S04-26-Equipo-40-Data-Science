/**
 * ChartFactory.js — ConversaAI Dashboard
 *
 * PATTERN: Factory Method
 * Creates chart instances without the consumer needing to know
 * which concrete renderer to use. Charts are selected by type string.
 *
 * SOLID — Open/Closed:
 * New chart types are registered with ChartFactory.register() without
 * modifying existing factory logic.
 *
 * SOLID — Dependency Inversion:
 * Consumers (UIController, app.js) depend on ChartFactory (abstraction),
 * not on concrete BarChart/LineChart implementations.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};

    /* ----------------------------------------------------------------
       Color palette (shared across all charts)
       ---------------------------------------------------------------- */
    const PALETTE = {
        indigo:  { solid: 'rgba(99, 102, 241, 0.85)',  border: 'rgba(99, 102, 241, 1)',   fill: 'rgba(99, 102, 241, 0.12)' },
        emerald: { solid: 'rgba(16, 185, 129, 0.85)',  border: 'rgba(16, 185, 129, 1)',   fill: 'rgba(16, 185, 129, 0.12)' },
        orange:  { solid: 'rgba(249, 115, 22,  0.85)', border: 'rgba(249, 115, 22,  1)',   fill: 'rgba(249, 115, 22,  0.12)' },
        rose:    { solid: 'rgba(244, 63,  94,  0.85)', border: 'rgba(244, 63,  94,  1)',   fill: 'rgba(244, 63,  94,  0.12)' },
        violet:  { solid: 'rgba(139, 92,  246, 0.85)', border: 'rgba(139, 92,  246, 1)',   fill: 'rgba(139, 92,  246, 0.12)' },
        cyan:    { solid: 'rgba(6,   182, 212, 0.85)', border: 'rgba(6,   182, 212, 1)',   fill: 'rgba(6,   182, 212, 0.12)' },
        amber:   { solid: 'rgba(245, 158, 11,  0.85)', border: 'rgba(245, 158, 11,  1)',   fill: 'rgba(245, 158, 11,  0.12)' },
        slate:   { solid: 'rgba(100, 116, 139, 0.7)',  border: 'rgba(100, 116, 139, 1)',   fill: 'rgba(100, 116, 139, 0.1)' }
    };

    /* ----------------------------------------------------------------
       Factory registry — OCP: register new types, never modify create()
       ---------------------------------------------------------------- */
    const _renderers = {
        bar:      () => ConversaAI.Charts.BarChart,
        line:     () => ConversaAI.Charts.LineChart,
        radar:    () => ConversaAI.Charts.RadarChart,
        doughnut: () => ConversaAI.Charts.DoughnutChart
    };

    const ChartFactory = {
        /**
         * Register a new chart type. OCP extension point.
         * @param {string}   type     - Chart type key
         * @param {Function} resolver - Returns the renderer object
         */
        register(type, resolver) {
            _renderers[type] = resolver;
        },

        /**
         * Create (or recreate) a chart.
         * @param {string}  type     - 'bar'|'line'|'radar'|'doughnut'
         * @param {string}  canvasId - Target canvas id
         * @param {Object}  config   - Renderer-specific config
         * @param {Object}  existing - Previous Chart.js instance (optional)
         * @returns {Chart|null}
         */
        create(type, canvasId, config, existing = null) {
            const resolver = _renderers[type];
            if (!resolver) {
                console.error(`[ChartFactory] Unknown chart type: "${type}"`);
                return null;
            }
            return resolver().render(canvasId, config, existing);
        },

        /** Expose palette for dataset construction */
        palette: Object.freeze(PALETTE),

        /* ----------------------------------------------------------------
           Pre-built config builders (keeps UIController clean)
           ---------------------------------------------------------------- */

        /** Build accuracy bar chart config from model array */
        buildAccuracyConfig(models) {
            const colorMap = ['slate', 'orange', 'emerald', 'indigo'];
            return {
                labels: models.map(m => m.name),
                datasets: [{
                    label: 'Precisión (%)',
                    data: models.map(m => m.accuracy),
                    backgroundColor: models.map((_, i) => PALETTE[colorMap[i] || 'indigo'].solid),
                    borderColor:     models.map((_, i) => PALETTE[colorMap[i] || 'indigo'].border),
                    borderWidth: 1,
                    borderRadius: 7,
                    borderSkipped: false
                }],
                options: {
                    scales: {
                        y: { min: 60, max: 100 }
                    }
                }
            };
        },

        /** Build latency bar chart config */
        buildLatencyConfig(models) {
            const colorMap = ['emerald', 'violet', 'cyan', 'rose'];
            return {
                labels: models.map(m => m.name),
                datasets: [{
                    label: 'Latencia (ms)',
                    data: models.map(m => m.latencyMs),
                    backgroundColor: models.map((_, i) => PALETTE[colorMap[i] || 'violet'].solid),
                    borderColor:     models.map((_, i) => PALETTE[colorMap[i] || 'violet'].border),
                    borderWidth: 1,
                    borderRadius: 7,
                    borderSkipped: false
                }],
                options: { scales: { y: { beginAtZero: true } } }
            };
        },

        /** Build radar config from model list */
        buildRadarConfig(models) {
            const colors = ['slate', 'orange', 'emerald', 'indigo'];
            return {
                labels: ['Precisión', 'Recall', 'F1-Score', 'Velocidad', 'Cobertura'],
                datasets: models.map((m, i) => {
                    const c = PALETTE[colors[i] || 'indigo'];
                    // Velocidad = inverse of latency (normalized 0-100)
                    const speed = Math.max(0, 100 - Math.round(m.latencyMs / 2));
                    // Cobertura: es+pt = 100, es or pt = 60
                    const coverage = m.lang === 'es+pt' ? 100 : 60;
                    return {
                        label: m.name,
                        data: [m.precision, m.recall, m.f1Score, speed, coverage],
                        backgroundColor: c.fill,
                        borderColor: c.border,
                        borderWidth: 2,
                        pointBackgroundColor: c.border,
                        pointBorderColor: '#fff',
                        pointRadius: 4,
                        pointHoverRadius: 6
                    };
                })
            };
        },

        /** Build F1 grouped bar config per sentiment class */
        buildF1Config(models) {
            const classColors = { positive: PALETTE.emerald, negative: PALETTE.rose, neutral: PALETTE.amber };
            return {
                labels: models.map(m => m.name),
                datasets: [
                    {
                        label: 'Positivo',
                        data: models.map(m => m.f1ByClass.positive),
                        backgroundColor: classColors.positive.solid,
                        borderColor: classColors.positive.border,
                        borderWidth: 1,
                        borderRadius: 5,
                        borderSkipped: false
                    },
                    {
                        label: 'Negativo',
                        data: models.map(m => m.f1ByClass.negative),
                        backgroundColor: classColors.negative.solid,
                        borderColor: classColors.negative.border,
                        borderWidth: 1,
                        borderRadius: 5,
                        borderSkipped: false
                    },
                    {
                        label: 'Neutro',
                        data: models.map(m => m.f1ByClass.neutral),
                        backgroundColor: classColors.neutral.solid,
                        borderColor: classColors.neutral.border,
                        borderWidth: 1,
                        borderRadius: 5,
                        borderSkipped: false
                    }
                ],
                options: {
                    plugins: { legend: { display: true } },
                    scales: { y: { min: 60, max: 100 } }
                }
            };
        },

        /** Build sentiment doughnut config */
        buildSentimentConfig(dist) {
            return {
                labels: ['Positivo', 'Neutral', 'Negativo'],
                datasets: [{
                    data: [dist.positive, dist.neutral, dist.negative],
                    backgroundColor: [PALETTE.emerald.solid, PALETTE.amber.solid, PALETTE.rose.solid],
                    borderColor: [PALETTE.emerald.border, PALETTE.amber.border, PALETTE.rose.border],
                    borderWidth: 2,
                    hoverOffset: 8
                }]
            };
        },

        /** Build intent horizontal bar config */
        buildIntentConfig(intents) {
            return {
                labels: intents.map(i => i.label),
                datasets: [{
                    label: 'Mensajes (%)',
                    data: intents.map(i => i.value),
                    backgroundColor: [
                        PALETTE.indigo.solid, PALETTE.rose.solid, PALETTE.cyan.solid,
                        PALETTE.orange.solid, PALETTE.violet.solid, PALETTE.emerald.solid
                    ],
                    borderColor: [
                        PALETTE.indigo.border, PALETTE.rose.border, PALETTE.cyan.border,
                        PALETTE.orange.border, PALETTE.violet.border, PALETTE.emerald.border
                    ],
                    borderWidth: 1,
                    borderRadius: 5
                }],
                horizontal: true,
                options: {
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { beginAtZero: true, max: 50 },
                        y: { ticks: { color: '#94a3b8', font: { size: 11 } } }
                    }
                }
            };
        },

        /** Build timeline line config */
        buildTimelineConfig(timeline) {
            return {
                labels: timeline.map(d => d.date),
                datasets: [
                    {
                        label: '🇪🇸 Español',
                        data: timeline.map(d => d.es),
                        borderColor: PALETTE.indigo.border,
                        backgroundColor: PALETTE.indigo.fill,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        borderWidth: 2
                    },
                    {
                        label: '🇧🇷 Portugués',
                        data: timeline.map(d => d.pt),
                        borderColor: PALETTE.emerald.border,
                        backgroundColor: PALETTE.emerald.fill,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        borderWidth: 2
                    }
                ]
            };
        },

        /** Build churn risk doughnut config */
        buildChurnConfig(churn) {
            return {
                labels: churn.labels,
                datasets: [{
                    data: churn.values,
                    backgroundColor: [
                        PALETTE.emerald.solid, PALETTE.amber.solid,
                        PALETTE.orange.solid, PALETTE.rose.solid
                    ],
                    borderColor: [
                        PALETTE.emerald.border, PALETTE.amber.border,
                        PALETTE.orange.border, PALETTE.rose.border
                    ],
                    borderWidth: 2,
                    hoverOffset: 8
                }]
            };
        },

        /** Build frustration by intent horizontal bar */
        buildFrustIntentConfig(data) {
            return {
                labels: data.labels,
                datasets: [{
                    label: 'Frustración Promedio',
                    data: data.scores.map(s => Math.round(s * 100)),
                    backgroundColor: data.scores.map(s => {
                        if (s >= 0.7) return PALETTE.rose.solid;
                        if (s >= 0.5) return PALETTE.orange.solid;
                        if (s >= 0.3) return PALETTE.amber.solid;
                        return PALETTE.emerald.solid;
                    }),
                    borderColor: data.scores.map(s => {
                        if (s >= 0.7) return PALETTE.rose.border;
                        if (s >= 0.5) return PALETTE.orange.border;
                        if (s >= 0.3) return PALETTE.amber.border;
                        return PALETTE.emerald.border;
                    }),
                    borderWidth: 1,
                    borderRadius: 5
                }],
                horizontal: true,
                options: {
                    plugins: { legend: { display: false } },
                    scales: { x: { beginAtZero: true, max: 100 } }
                }
            };
        },

        /** Build frustration timeline line config */
        buildFrustTimelineConfig(timeline) {
            return {
                labels: timeline.map(d => d.date),
                datasets: [
                    {
                        label: '🇪🇸 Español',
                        data: timeline.map(d => d.es !== null ? Math.round(d.es * 100) : null),
                        borderColor: PALETTE.violet.border,
                        backgroundColor: PALETTE.violet.fill,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        borderWidth: 2,
                        spanGaps: true
                    },
                    {
                        label: '🇧🇷 Portugués',
                        data: timeline.map(d => d.pt !== null ? Math.round(d.pt * 100) : null),
                        borderColor: PALETTE.orange.border,
                        backgroundColor: PALETTE.orange.fill,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        borderWidth: 2,
                        spanGaps: true
                    }
                ],
                options: {
                    scales: {
                        y: { min: 0, max: 100, ticks: {
                            callback: v => v + '%'
                        }}
                    }
                }
            };
        }
    };

    ConversaAI.ChartFactory = ChartFactory;

})(window);
