/**
 * EventBus.js — ConversaAI Dashboard
 *
 * PATTERN: Observer (Pub/Sub)
 * Allows decoupled communication between modules.
 * Modules subscribe to events without direct references to each other.
 *
 * SOLID — Interface Segregation:
 * Only exposes on(), off(), emit(). Nothing else.
 */
(function (global) {
    'use strict';

    const ConversaAI = global.ConversaAI = global.ConversaAI || {};

    /**
     * EventBus — centralized event dispatcher.
     * All inter-module communication goes through here.
     */
    const EventBus = (function () {
        /** @type {Object.<string, Function[]>} */
        const _listeners = {};

        return {
            /**
             * Subscribe to an event.
             * @param {string}   event    - Event name
             * @param {Function} callback - Handler function
             */
            on(event, callback) {
                if (!_listeners[event]) _listeners[event] = [];
                _listeners[event].push(callback);
            },

            /**
             * Unsubscribe a specific handler from an event.
             * @param {string}   event    - Event name
             * @param {Function} callback - Handler to remove
             */
            off(event, callback) {
                if (!_listeners[event]) return;
                _listeners[event] = _listeners[event].filter(fn => fn !== callback);
            },

            /**
             * Emit an event, calling all subscribers with optional data.
             * @param {string} event - Event name
             * @param {*}      data  - Payload passed to handlers
             */
            emit(event, data) {
                if (!_listeners[event]) return;
                _listeners[event].forEach(fn => {
                    try { fn(data); }
                    catch (err) { console.error(`[EventBus] Error in "${event}" handler:`, err); }
                });
            }
        };
    })();

    ConversaAI.EventBus = EventBus;

    // Freeze to prevent accidental mutation
    Object.freeze(ConversaAI.EventBus);

})(window);
