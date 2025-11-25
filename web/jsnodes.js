// ImageStringBatchMulti 仅用按钮更新输入口
const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "lian.ImageStringBatchMulti",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const TARGET_NAMES = [
            "ImageStringBatchMulti",
            "lian 图像字符串批次构建·动态",
        ];
        if (!TARGET_NAMES.includes(nodeData.name)) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated ? onCreated.apply(this, arguments) : undefined;

            // 初始 4 对
            this._type_img = "IMAGE";
            this._type_str = "STRING";
            this._ensurePairs(4);

            // 仅此按钮：点击才刷新
            this.addWidget("button", "Update inputs", null, () => {
                const count = this.widgets.find(w => w.name === "inputcount")?.value || 1;
                this._ensurePairs(count);
                app.graph.setDirtyCanvas(true, true);
            });
            return r;
        };

        // 成对增删核心逻辑
        nodeType.prototype._ensurePairs = function (n) {
            if (!this.inputs) this.inputs = [];
            let pairs = 0;
            while (this.inputs.find(inp => inp.name === `image_${pairs + 1}`)) pairs++;
            if (pairs === n) return;

            if (pairs > n) {
                for (let i = pairs; i > n; --i) {
                    const idxStr = this.inputs.findIndex(inp => inp.name === `string_${i}`);
                    const idxImg = this.inputs.findIndex(inp => inp.name === `image_${i}`);
                    if (idxStr >= 0) this.removeInput(idxStr);
                    if (idxImg >= 0) this.removeInput(idxImg);
                }
            } else {
                for (let i = pairs + 1; i <= n; ++i) {
                    this.addInput(`image_${i}`, this._type_img, { shape: 7 });
                    this.addInput(`string_${i}`, this._type_str, { shape: 7 });
                }
            }
        };
    }
});