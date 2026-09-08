import { expect, test } from "@playwright/test";

for (const width of [1100, 390, 320]) {
  test(`completion editor stays still through typing, deletion, and selection at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("http://127.0.0.1:4176/app?layoutFixture=base");
    const editor = page.getByRole("textbox", { name: "Editable completion buffer" });
    await editor.fill("Once beneath the moon: ");
    await page.getByRole("button", { name: "Continue text", exact: true }).click();
    await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeDisabled();
    await editor.focus();
    await editor.evaluate(element => {
      const input = element as HTMLTextAreaElement;
      input.setSelectionRange(input.value.length, input.value.length);
      input.dispatchEvent(new Event("select", { bubbles: true }));
    });
    const bounds = (await editor.boundingBox())!;
    const stable = async () => {
      const current = (await editor.boundingBox())!;
      for (const dimension of ["x", "y", "width", "height"] as const) {
        expect(Math.abs(current[dimension] - bounds[dimension]), dimension).toBeLessThanOrEqual(1);
      }
    };
    for (const key of ["Backspace", "a", "b", "Enter", "Backspace"]) {
      await page.keyboard.down(key);
      await stable();
      await page.keyboard.up(key);
      await stable();
    }
    await editor.evaluate(element => {
      const input = element as HTMLTextAreaElement;
      input.setSelectionRange(0, input.value.length);
      input.dispatchEvent(new Event("select", { bubbles: true }));
    });
    await stable();
    await page.keyboard.press("Backspace");
    await expect(editor).toHaveValue("");
    await stable();
    await page.keyboard.insertText("A long editable line.\n".repeat(80));
    await stable();
    expect(await editor.evaluate(el => el.scrollHeight > el.clientHeight)).toBe(true);
    const actions = page.getByRole("group", { name: "Completion actions", exact: true });
    await expect(actions.getByRole("button", { name: "Continue from cursor", exact: true })).toBeVisible();
    const cursor = (await actions.getByRole("button", { name: "Continue from cursor", exact: true }).boundingBox())!;
    const end = (await actions.getByRole("button", { name: "Continue text", exact: true }).boundingBox())!;
    expect(Math.abs(cursor.y - end.y)).toBeLessThanOrEqual(1);
    expect(cursor.y).toBeGreaterThanOrEqual((await editor.boundingBox())!.y + bounds.height);
    expect(await actions.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    await page.screenshot({ path: info.outputPath(`stable-editor-${width}.png`) });
  });
}
