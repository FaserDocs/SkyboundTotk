// The browser build's JS side (linked with --js-library): the renderer's canvas lives in its own worker (an
// OffscreenCanvas made there); each presented frame goes to the page as an ImageBitmap.
addToLibrary({
  rx_web_canvas_init__deps: ['$specialHTMLTargets'],
  rx_web_canvas_init: (w, h) => {
    var c = new OffscreenCanvas(w, h);
    specialHTMLTargets['#rx'] = c;
    Module['rxCanvas'] = c;
  },
  rx_web_canvas_resize: (w, h) => {
    var c = Module['rxCanvas'];
    if (c && (c.width != w || c.height != h)) {
      c.width = w;
      c.height = h;
    }
  },
  rx_web_present: () => {
    if (typeof document == 'undefined') postMessage({cmd: {{{ CMD_CALL_HANDLER }}}, handler: 'rxFrame', args: []});
  },
});
