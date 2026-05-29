/**
 * RadarChart.js — ConversaAI Dashboard
 * SOLID — Liskov Substitution + Single Responsibility
 * Concrete renderer for radar/spider charts. Same render() interface.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};
    ConversaAI.Charts = ConversaAI.Charts || {};

    function _baseOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 900, easing: 'easeOutBack' },
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
                    cornerRadius: 10
                }
            },
            scales: {
                r: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        color: '#4b5675',
                        backdropColor: 'transparent',
                        font: { size: 9 }
                    },
                    grid: { color: 'rgba(255,255,255,0.07)' },
                    pointLabels: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter', weight: '500' }
                    },
                    angleLines: { color: 'rgba(255,255,255,0.07)' }
                }
            }
        };
    }

    const RadarChart = {
        /**
         * Render (or update) a radar chart.
         * @param {string}  canvasId - Canvas element id
         * @param {Object}  config   - { labels, datasets, options? }
         * @param {Object}  existing - Existing Chart.js instance (optional)
         * @returns {Chart}
         */
        render(canvasId, config, existing) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) { console.warn(`[RadarChart] Canvas #${canvasId} not found`); return null; }
            if (existing) { existing.destroy(); }

            const opts = Object.assign({}, _baseOptions(), config.options || {});

            return new Chart(canvas.getContext('2d'), {
                type: 'radar',
                data: { labels: config.labels, datasets: config.datasets },
                options: opts
            });
        }
    };

    ConversaAI.Charts.RadarChart = Object.freeze(RadarChart);

})(window);
