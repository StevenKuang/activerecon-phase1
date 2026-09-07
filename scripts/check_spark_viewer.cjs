#!/usr/bin/env node
/* Browser acceptance test for the curated live Spark catalog.
 * Requires Playwright (npm install --no-save playwright), plus Chromium.
 * node scripts/check_spark_viewer.cjs http://127.0.0.1:8090 outputs/spark-audit
 * SPARK_CHROMIUM may select a full Chromium executable for hardware EGL.
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const base = (process.argv[2] || 'http://127.0.0.1:8090').replace(/\/$/, '');
const out = path.resolve(process.argv[3] || 'outputs/spark-audit');
fs.mkdirSync(out, { recursive: true });

function query(view) {
  return new URLSearchParams({round: view.round, scene: view.scene,
    a_difficulty: view.difficulty, a_method: view.method,
    a_reconstruction: view.reconstruction, a_seed: view.seed});
}

async function ready(page, methods) {
  await page.waitForFunction((expected) => {
    const d = window.__abDebug;
    if (!d || !document.getElementById('loading').hidden || d.runs.length !== expected.length) return false;
    return d.runs.every((r, i) => r.rm.method === expected[i]
      && [...r.splatMeshes.values()].some(m => m.userData.splatReady && m.visible && m.numSplats > 0));
  }, methods, {timeout: 180000});
  // Allow LoD pages and the asynchronous sorter to catch up after initialization.
  await page.waitForTimeout(1800);
  const state = await page.evaluate(() => {
    const d = window.__abDebug, gl = d.renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return {gpu: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      pagerMismatches: d.pagerMismatches(), pendingBuffers: d.pendingBinaryRequests,
      drawCalls: d.renderer.info.render.calls,
      compareVisible: getComputedStyle(document.getElementById('paneB-row')).display !== 'none',
      runs: d.runs.map(r => ({method:r.rm.method, frames:r.frames.length, tEnd:r.tEnd,
        models:[...r.splatMeshes].map(([name,m]) => ({name,count:m.numSplats,ready:m.userData.splatReady}))}))};
  });
  assert.deepEqual(state.pagerMismatches, []);
  assert.equal(state.pendingBuffers, 0, 'completed replay buffers retained in request cache');
  assert.ok(state.drawCalls > 0, 'no WebGL draw calls');
  assert.equal(state.compareVisible, methods.length > 1, 'wrong comparison control visibility');
  return state;
}

async function seek(page, fraction) {
  await page.locator('#progress').evaluate((slider, value) => {
    slider.value = String(value);
    slider.dispatchEvent(new Event('input', {bubbles:true}));
  }, fraction);
}

(async () => {
  const index = await (await fetch(`${base}/cache-index.json`)).json();
  const catalog = [...new Map(index.artifacts.filter(a => a.kind === 'reconstruction')
    .flatMap(a => a.selections).map(v => [JSON.stringify(v), v])).values()];
  const only = process.env.SPARK_TEST_SELECTIONS?.split(',');
  const id = v => `${v.scene}__${v.difficulty}__${v.seed}__${v.method}`;
  if (only) catalog.splice(0, catalog.length, ...catalog.filter(v =>
    only.includes(id(v))).sort((a,b) => only.indexOf(id(a))-only.indexOf(id(b))));
  assert.ok(catalog.length, 'empty reconstruction catalog');
  const browser = await chromium.launch({headless:true,
    executablePath:process.env.SPARK_CHROMIUM || undefined,
    args:['--no-sandbox','--use-gl=angle','--use-angle=gl-egl','--enable-gpu',
      '--ignore-gpu-blocklist','--disable-software-rasterizer']});
  const context = await browser.newContext({viewport:{width:1100,height:720},ignoreHTTPSErrors:true});
  const page = await context.newPage();
  let issues = [];
  page.on('pageerror', e => issues.push(e.message));
  page.on('console', m => { if (m.type() === 'error' && !m.location().url.endsWith('/favicon.ico')) issues.push(m.text()); });
  page.on('response', r => {if (r.status() >= 400 && !r.url().endsWith('/favicon.ico')) issues.push(`${r.status()} ${r.url()}`);});
  const report = {catalog, views:[], interactions:[], errors:[]};
  try {
    for (const view of catalog) {
      issues = [];
      const id = `${view.scene}__${view.difficulty}__${view.seed}__${view.method}`;
      try {
        await page.goto(`${base}/view?${query(view)}`, {waitUntil:'domcontentloaded',timeout:180000});
        const state = await ready(page,[view.method]);
        const atCapture = await page.evaluate(() => {
          const d = window.__abDebug;
          return d.camera.position.distanceTo(d.runs[0].framePos[0]) < 1e-5;
        });
        assert.ok(atCapture, 'initial camera must use the recorded start pose');
        await seek(page, 0.5);
        if (await page.evaluate(() => window.__abDebug.state.playing)) await page.locator('#play').click();
        await page.waitForTimeout(500);
        await page.screenshot({path:path.join(out, `${id}.png`)});
        assert.equal(issues.length,0,issues.join('\n'));
        report.views.push({id,ok:true,...state});
        console.log(`PASS ${id} (${state.runs[0].frames} frames)`);
      } catch (e) {
        report.views.push({id,ok:false,error:e.message,issues:[...issues]});
        report.errors.push(id);
        console.error(`FAIL ${id}: ${e.message}`);
      }
    }
    // Churn methods in a single SPA: the same camera/time and pane pager must survive.
    const first = catalog[0];
    await page.goto(`${base}/view?${query(first)}`, {waitUntil:'domcontentloaded'});
    await ready(page,[first.method]);
    if (await page.evaluate(() => window.__abDebug.state.playing)) await page.locator('#play').click();
    issues = [];
    await seek(page, 0.4);
    const before = await page.evaluate(() => ({t:window.__abDebug.state.t, camera:window.__abDebug.camera.position.toArray()}));
    const sameScene = catalog.filter(v => v.scene===first.scene && v.difficulty===first.difficulty);
    for (const view of [...sameScene, first]) {
      await page.selectOption('#method-a',view.method);
      await page.locator('#live-apply').click();
      await ready(page,[view.method]);
      const after = await page.evaluate(() => ({t:window.__abDebug.state.t, camera:window.__abDebug.camera.position.toArray(),playing:window.__abDebug.state.playing}));
      assert.ok(Math.abs(after.t-before.t)<0.05,'method switch reset playback time');
      assert.deepEqual(after.camera,before.camera,'method switch reset camera');
      assert.equal(after.playing,false);
    }
    report.interactions.push('method churn preserves time, pause, camera and pager ownership');
    await page.locator('#next').click();
    const stepped = await page.evaluate(() => window.__abDebug.state.t);
    assert.ok(stepped > before.t, 'next capture did not advance');
    await page.locator('#prev').click();
    assert.ok(await page.evaluate(t => window.__abDebug.state.t < t, stepped), 'previous capture did not go back');
    await page.locator('#capture-view').click();
    await page.waitForTimeout(100);
    assert.ok(await page.evaluate(() => {
      const d = window.__abDebug, r = d.runs[0];
      return d.camera.position.distanceTo(r.framePos[r.captureIndexAt(d.state.t)]) < 1e-5
        && !r.markerImg.visible;
    }), 'go-to-capture did not position the camera');
    await page.locator('#layer-splat').uncheck();
    assert.ok(await page.evaluate(() => !window.__abDebug.state.layers.splat));
    await page.locator('#layer-splat').check();
    await ready(page,[first.method]);
    report.interactions.push('capture stepping, recorded-camera reset and point-cloud/splat toggle');
    // Inspect the comparison from the common start pose, not a planner's
    // potentially wall-adjacent mid-trajectory pose.
    await seek(page, 0);
    await page.locator('#capture-view').click();
    await seek(page, 0.4);
    await page.locator('#compare').check();
    await ready(page,[first.method,first.method]);
    const other = sameScene.find(v=>v.method!==first.method);
    await page.selectOption('#method-b',other.method);
    await page.locator('#live-apply').click();
    await ready(page,[first.method,other.method]);
    await page.screenshot({path:path.join(out,'compare.png')});
    await page.locator('#compare').uncheck();
    await ready(page,[first.method]);
    assert.equal(issues.length,0,issues.join('\n'));
    report.interactions.push('independent comparison panes and return to single view');
    console.log('PASS SPA churn and comparison');
  } catch(e) {
    report.errors.push(`interaction: ${e.message}`);
    console.error(e);
  } finally {
    fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify(report,null,2)+'\n');
    await browser.close();
  }
  if (report.errors.length) process.exitCode=1;
})();
