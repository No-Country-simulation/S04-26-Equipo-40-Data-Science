/**
 * DoughnutChart.js — ConversaAI Dashboard
 * SOLID — Liskov Substitution + Single Responsibility
 * Concrete renderer for doughnut/pie charts. Same render() interface.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};
    ConversaAI.Charts = ConversaAI.Charts || {};

    function _baseOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800, easing: 'easeOutQuart' },
            cutout: '68%',
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter' },
                        boxWidth: 12, boxHeight: 12,
                        borderRadius: 3, padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(13, 20, 38, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 10,
                    callbacks: {
                        label(ctx) {
                            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = ((ctx.parsed / total) * 100).toFixed(1);
                            return ` ${ctx.label}: ${pct}%`;
                        }
                    }
                }
            }
        };
    }

    const DoughnutChart = {
        /**
         * Render (or update) a doughnut chart.
         * @param {string}  canvasId - Canvas element id
         * @param {Object}  config   - { labels, datasets, options? }
         * @param {Object}  existing - Existing Chart.js instance (optional)
         * @returns {Chart}
         */
        render(canvasId, config, existing) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) { console.warn(`[DoughnutChart] Canvas #${canvasId} not found`); return null; }
            if (existing) { existing.destroy(); }

            const opts = Object.assign({}, _baseOptions(), config.options || {});

            return new Chart(canvas.getContext('2d'), {
                type: 'doughnut',
                data: { labels: config.labels, datasets: config.datasets },
                options: opts
            });
        }
    };

    ConversaAI.Charts.DoughnutChart = Object.freeze(DoughnutChart);

})(window);
