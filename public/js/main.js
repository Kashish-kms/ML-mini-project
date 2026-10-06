/**
 * main.js — Frontend logic for CarValueAI
 *
 * - Populates dropdowns from /metadata
 * - Submits form to /predict
 * - Renders results, deal badge, and Chart.js factor chart
 * - Populates Insights tables from /insights
 */

document.addEventListener("DOMContentLoaded", () => {
    // ── Tab navigation ────────────────────────────────────
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;

            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            tabContents.forEach(tc => {
                tc.classList.remove("active");
                if (tc.id === `tab-content-${target}`) {
                    tc.classList.add("active");
                }
            });

            // Load insights on first click
            if (target === "insights" && !insightsLoaded) {
                loadInsights();
            }
        });
    });

    // ── Populate dropdowns from metadata ──────────────────
    fetch("/metadata")
        .then(r => r.json())
        .then(meta => {
            const catOpts = meta.categorical_options || {};

            populateSelect("brand", catOpts.brand || []);
            populateSelect("fuel", catOpts.fuel || []);
            populateSelect("seller_type", catOpts.seller_type || []);
            populateSelect("transmission", catOpts.transmission || []);
            populateSelect("owner", catOpts.owner || []);
        })
        .catch(err => {
            console.warn("Could not load metadata:", err);
        });

    function populateSelect(id, options) {
        const sel = document.getElementById(id);
        if (!sel) return;
        options.forEach(opt => {
            const el = document.createElement("option");
            el.value = opt;
            el.textContent = opt;
            sel.appendChild(el);
        });
    }

    // ── Form submission ───────────────────────────────────
    const form = document.getElementById("predict-form");
    const btnText = document.querySelector(".btn-text");
    const btnLoader = document.getElementById("btn-loader");
    const resultCard = document.getElementById("result-card");
    const errorToast = document.getElementById("error-toast");
    const errorMsg = document.getElementById("error-msg");
    const errorClose = document.getElementById("error-close");

    let factorsChart = null;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        hideError();

        // Show loading
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");

        const yearVal = parseInt(document.getElementById("year").value);
        const carAge = 2024 - yearVal;

        const payload = {
            brand:        document.getElementById("brand").value,
            car_age:      carAge,
            km_driven:    parseFloat(document.getElementById("km_driven").value),
            fuel:         document.getElementById("fuel").value,
            seller_type:  document.getElementById("seller_type").value,
            transmission: document.getElementById("transmission").value,
            owner:        document.getElementById("owner").value,
            engine:       parseFloat(document.getElementById("engine").value),
            max_power:    parseFloat(document.getElementById("max_power").value),
            mileage:      parseFloat(document.getElementById("mileage").value),
            seats:        parseFloat(document.getElementById("seats").value),
        };

        const listedPrice = document.getElementById("listed_price").value;
        if (listedPrice && listedPrice.trim() !== "") {
            payload.listed_price = parseFloat(listedPrice);
        }

        try {
            const resp = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.error || "Prediction failed");
            }

            // Display results
            renderResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            btnText.classList.remove("hidden");
            btnLoader.classList.add("hidden");
        }
    });

    function renderResult(data) {
        resultCard.classList.remove("hidden");

        // Smooth scroll on mobile
        resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

        // Price
        const priceEl = document.getElementById("result-price");
        priceEl.textContent = `₹ ${data.predicted_price_lakh} Lakh`;

        const rangeEl = document.getElementById("result-range");
        rangeEl.textContent = `Range: ₹ ${data.price_range.low_lakh} — ${data.price_range.high_lakh} Lakh`;

        // Deal badge
        const badge = document.getElementById("deal-badge");
        const dealIcon = document.getElementById("deal-icon");
        const dealText = document.getElementById("deal-text");
        const dealConf = document.getElementById("deal-confidence");

        badge.className = `deal-badge ${data.deal_color}`;
        dealText.textContent = data.deal_label;

        const icons = { Underpriced: "🟢", Fair: "🟡", Overpriced: "🔴" };
        dealIcon.textContent = icons[data.deal_label] || "⚪";

        dealConf.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

        // Factors chart
        const factors = data.top_factors || [];
        if (factors.length > 0) {
            document.getElementById("factors-section").classList.remove("hidden");
            renderFactorsChart(factors);
        }
    }

    function renderFactorsChart(factors) {
        const ctx = document.getElementById("factors-chart").getContext("2d");

        if (factorsChart) {
            factorsChart.destroy();
        }

        const labels = factors.map(f => cleanFeatureName(f.feature));
        const values = factors.map(f => f.impact);
        const colors = values.map(v => v >= 0 ? "#10b981" : "#ef4444");

        factorsChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "SHAP Impact",
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 6,
                    borderSkipped: false,
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#1a1f35",
                        titleColor: "#f1f5f9",
                        bodyColor: "#94a3b8",
                        borderColor: "#2d3555",
                        borderWidth: 1,
                        cornerRadius: 8,
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(45, 53, 85, 0.5)" },
                        ticks: { color: "#94a3b8" },
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: "#f1f5f9", font: { weight: 500 } },
                    }
                }
            }
        });
    }

    function cleanFeatureName(name) {
        // Convert encoded feature names to readable
        return name
            .replace(/^cat__/, "")
            .replace(/^num__/, "")
            .replace(/_/g, " ")
            .replace(/\b\w/g, c => c.toUpperCase());
    }

    // ── Error handling ────────────────────────────────────
    function showError(msg) {
        errorMsg.textContent = msg;
        errorToast.classList.remove("hidden");
        setTimeout(() => hideError(), 8000);
    }

    function hideError() {
        errorToast.classList.add("hidden");
    }

    errorClose.addEventListener("click", hideError);

    // ── Insights Tab ──────────────────────────────────────
    let insightsLoaded = false;

    function loadInsights() {
        fetch("/insights")
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    console.warn("Insights error:", data.error);
                    return;
                }

                // Regression table
                const regBody = document.getElementById("reg-table-body");
                const regComp = data.regression?.comparison || [];
                regComp.forEach(row => {
                    const tr = document.createElement("tr");
                    if (row.model.includes("tuned")) tr.classList.add("best-row");
                    tr.innerHTML = `
                        <td>${row.model}</td>
                        <td>₹${numberWithCommas(row.MAE)}</td>
                        <td>₹${numberWithCommas(row.RMSE)}</td>
                        <td>${row.R2}</td>
                    `;
                    regBody.appendChild(tr);
                });

                // Classification table
                const clfBody = document.getElementById("clf-table-body");
                const clfComp = data.classification?.comparison || [];
                clfComp.forEach(row => {
                    const tr = document.createElement("tr");
                    if (row.model.includes("tuned")) tr.classList.add("best-row");
                    tr.innerHTML = `
                        <td>${row.model}</td>
                        <td>${row.accuracy}</td>
                        <td>${row.precision}</td>
                        <td>${row.recall}</td>
                        <td>${row.f1}</td>
                        <td>${row.roc_auc}</td>
                    `;
                    clfBody.appendChild(tr);
                });

                insightsLoaded = true;
            })
            .catch(err => console.error("Failed to load insights:", err));
    }

    function numberWithCommas(x) {
        return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
});
