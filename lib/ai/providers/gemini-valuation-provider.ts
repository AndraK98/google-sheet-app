import { extractGrounding, runWithGeminiModelFallback } from "@/lib/ai/google-genai";
import { type ValuationCatalogContext, type ValuationProvider } from "@/lib/ai/types";
import { getComplexityMultiplier } from "@/lib/services/quote-service";
import { type Setting, type Stone, type ValuationEstimate, type ValuationMessage, type ValuationRequestInput } from "@/lib/types";
import { valuationEstimateSchema } from "@/lib/validators";

function parseDataUrl(dataUrl: string): { mimeType: string; data: string } | null {
  const match = dataUrl.match(/^data:(.+?);base64,(.+)$/);
  if (!match) {
    return null;
  }

  return {
    mimeType: match[1],
    data: match[2],
  };
}

function extractJsonText(raw: string) {
  const trimmed = raw.trim();

  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    return trimmed;
  }

  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]+?)\s*```/i);
  if (fencedMatch?.[1]) {
    return fencedMatch[1].trim();
  }

  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");

  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    return trimmed.slice(firstBrace, lastBrace + 1);
  }

  return trimmed;
}

function normalizeValuationTarget(value: unknown) {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase();

  if (!normalized) {
    return "piece";
  }

  if (normalized === "stone" || normalized.includes("stone")) {
    return "stone";
  }

  if (
    normalized === "setting" ||
    normalized.includes("setting") ||
    normalized.includes("mount") ||
    normalized.includes("band")
  ) {
    return "setting";
  }

  return "piece";
}

function normalizeNumericValue(value: unknown) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : 0;
  }

  const normalized = String(value ?? "").trim();

  if (!normalized) {
    return 0;
  }

  const direct = Number(normalized.replace(/,/g, ""));
  if (Number.isFinite(direct)) {
    return direct;
  }

  const match = normalized.replace(/,/g, "").match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return 0;
  }

  const parsed = Number(match[0]);
  return Number.isFinite(parsed) ? parsed : 0;
}

function normalizeTextValue(value: unknown, fallback = "") {
  const normalized = String(value ?? "").trim();
  return normalized || fallback;
}

function roundMoney(value: number) {
  return Math.round(value * 100) / 100;
}

function clampComplexity(value: number) {
  const normalized = Math.round(value);

  if (normalized < 1) {
    return 0;
  }

  if (normalized > 5) {
    return 5;
  }

  return normalized;
}

function normalizeEstimateRange(low: number, high: number) {
  const safeLow = Number.isFinite(low) && low >= 0 ? low : 0;
  const safeHigh = Number.isFinite(high) && high >= 0 ? high : 0;

  if (safeLow > 0 && safeHigh > 0) {
    return safeLow <= safeHigh ? { low: safeLow, high: safeHigh } : { low: safeHigh, high: safeLow };
  }

  if (safeHigh > 0) {
    return { low: roundMoney(safeHigh * 0.9), high: safeHigh };
  }

  if (safeLow > 0) {
    return { low: safeLow, high: roundMoney(safeLow * 1.1) };
  }

  return { low: 0, high: 0 };
}

function inferMetalFromText(description: string) {
  const normalized = description.toLowerCase();

  if (normalized.includes("platinum")) {
    return "Platinum";
  }

  if (normalized.includes("silver")) {
    return "Silver";
  }

  if (normalized.includes("18k")) {
    return "18K Gold";
  }

  if (normalized.includes("14k")) {
    return "14K Gold";
  }

  if (normalized.includes("gold")) {
    return "Gold";
  }

  return "";
}

function inferWeightFromText(description: string) {
  const normalized = description.replace(/,/g, ".");
  const match = normalized.match(/(\d+(?:\.\d+)?)\s*(?:g|gr|gram|grams)\b/i);

  if (!match) {
    return 0;
  }

  const parsed = Number(match[1]);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
}

function inferCaratFromText(description: string) {
  const normalized = description.replace(/,/g, ".");
  const match = normalized.match(/(\d+(?:\.\d+)?)\s*(?:ct|carat|carats)\b/i);

  if (!match) {
    return 0;
  }

  const parsed = Number(match[1]);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
}

function tokenizeDescription(description: string) {
  return Array.from(
    new Set(
      description
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter((token) => token.length >= 3),
    ),
  );
}

function hasStoneIntent(description: string) {
  return /\b(stone|diamond|sapphire|ruby|emerald|moissanite|opal|garnet|topaz|tourmaline|amethyst|aquamarine|morganite|spinel|onyx|pearl|gem|gems|carat|carats|lab diamond|lab dia)\b/i.test(
    description,
  );
}

function buildWeightedTokens(description: string) {
  const genericTokens = new Set([
    "ring",
    "gold",
    "size",
    "pure",
    "yellow",
    "white",
    "rose",
    "metal",
    "piece",
    "with",
    "for",
    "and",
    "the",
    "this",
  ]);

  return tokenizeDescription(description).map((token) => ({
    token,
    weight: genericTokens.has(token) ? 1 : 4,
  }));
}

function inferComplexityFromText(description: string) {
  const normalized = description.toLowerCase();

  if (/\bsignet\b/.test(normalized)) {
    return 1;
  }

  if (/\b(plain|simple|minimal|solitaire|band)\b/.test(normalized)) {
    return 1;
  }

  if (/\b(curve|knife edge|bezel|twist|double band)\b/.test(normalized)) {
    return 2;
  }

  if (/\b(halo|cluster|three stone|trilogy|accent)\b/.test(normalized)) {
    return 3;
  }

  if (/\b(pave|micro pave|hidden halo|split shank|crown)\b/.test(normalized)) {
    return 4;
  }

  if (/\b(vintage|intricate|engraved|filigree|cathedral|statement)\b/.test(normalized)) {
    return 5;
  }

  return 0;
}

function resolveMetalRate(metal: string, context: ValuationCatalogContext) {
  const normalized = metal.toLowerCase();

  if (normalized.includes("silver")) {
    return context.defaults.metalPrices.silver;
  }

  if (normalized.includes("platinum")) {
    return context.defaults.metalPrices.platinum;
  }

  return context.defaults.metalPrices.gold;
}

function findBestCatalogMatch<T>(
  items: T[],
  haystackBuilder: (item: T) => string,
  description: string,
): T | undefined {
  const weightedTokens = buildWeightedTokens(description);

  if (!weightedTokens.length) {
    return undefined;
  }

  let bestScore = 0;
  let bestItem: T | undefined;

  for (const item of items) {
    const haystack = haystackBuilder(item).toLowerCase();
    let score = 0;

    for (const { token, weight } of weightedTokens) {
      if (haystack.includes(token)) {
        score += weight;
      }
    }

    if (score > bestScore) {
      bestScore = score;
      bestItem = item;
    }
  }

  return bestScore >= 4 ? bestItem : undefined;
}

function fallbackMatchedStone(context: ValuationCatalogContext, description: string, parsed: unknown) {
  const source = parsed as Record<string, unknown>;
  const matchedStoneId = normalizeTextValue(source.matched_catalog_stone_id);

  if (matchedStoneId) {
    const directMatch = context.stones.find((stone) => stone.stone_id.trim().toUpperCase() === matchedStoneId.toUpperCase());
    if (directMatch) {
      return directMatch;
    }
  }

  if (!hasStoneIntent(description)) {
    return undefined;
  }

  return findBestCatalogMatch(
    context.stones,
    (stone) => `${stone.stone_id} ${stone.name} ${stone.shape} ${stone.color} ${stone.quality}`,
    description,
  );
}

function fallbackMatchedSetting(context: ValuationCatalogContext, description: string, parsed: unknown) {
  const source = parsed as Record<string, unknown>;
  const matchedSettingId = normalizeTextValue(source.matched_catalog_setting_id);

  if (matchedSettingId) {
    const directMatch = context.settings.find(
      (setting) => setting.setting_id.trim().toUpperCase() === matchedSettingId.toUpperCase(),
    );
    if (directMatch) {
      return directMatch;
    }
  }

  return findBestCatalogMatch(
    context.settings,
    (setting) => `${setting.setting_id} ${setting.style} ${setting.metal} ${setting.dimensions_mm} ${setting.stone_capacity}`,
    description,
  );
}

function estimateFromContext(
  input: ValuationRequestInput,
  context: ValuationCatalogContext,
  partial: ValuationEstimate,
  matchedStone: Stone | undefined,
  matchedSetting: Setting | undefined,
): ValuationEstimate {
  const descriptionHasStone = hasStoneIntent(input.description);
  const inferredMetal = normalizeTextValue(partial.inferred_metal) || inferMetalFromText(input.description) || matchedSetting?.metal || "";
  const inferredWeight =
    normalizeNumericValue(partial.inferred_gold_weight_g) || inferWeightFromText(input.description) || matchedSetting?.gold_weight_g || 0;
  const inferredCarat =
    descriptionHasStone
      ? normalizeNumericValue(partial.inferred_carat) || inferCaratFromText(input.description) || matchedStone?.carat || 0
      : 0;
  const inferredComplexity =
    clampComplexity(
      matchedSetting?.complexity_level ||
        normalizeNumericValue(partial.inferred_complexity_level) ||
        inferComplexityFromText(input.description) ||
        0,
    ) || 3;
  const stoneBase = descriptionHasStone ? matchedStone?.final_price ?? 0 : 0;
  const settingBase = matchedSetting?.base_price ?? 0;
  const laborBase = matchedSetting?.labor_cost ?? (inferredComplexity > 0 ? inferredComplexity * 40 : 0);
  const metalRate = resolveMetalRate(inferredMetal || matchedSetting?.metal || "gold", context);
  const materialBase = inferredWeight > 0 ? inferredWeight * metalRate : 0;
  const stoneTotal = descriptionHasStone
    ? roundMoney(normalizeNumericValue(partial.estimated_stone_total) || stoneBase)
    : 0;
  let settingTotal = roundMoney(normalizeNumericValue(partial.estimated_setting_total) || settingBase);

  if (settingTotal <= 0) {
    settingTotal = roundMoney(materialBase + laborBase);
  }

  if (settingTotal <= 0 && matchedSetting) {
    settingTotal = roundMoney(matchedSetting.base_price);
  }

  const complexityMultiplier = getComplexityMultiplier(inferredComplexity);
  const formulaTotal = roundMoney((stoneTotal + settingTotal) * complexityMultiplier);
  const currentRange = normalizeEstimateRange(partial.estimated_value_low, partial.estimated_value_high);
  const range =
    formulaTotal > 0
      ? { low: formulaTotal, high: formulaTotal }
      : currentRange.high > 0
      ? currentRange
      : {
          low: 0,
          high: 0,
        };
  const reasoning = normalizeTextValue(partial.reasoning);
  const recommendedNextStep = normalizeTextValue(partial.recommended_next_step);
  const canonicalPricingSummary = `Stone subtotal ${roundMoney(stoneTotal)} USD. Setting subtotal ${roundMoney(settingTotal)} USD. Complexity ${inferredComplexity} uses ${complexityMultiplier}x. Formula total ${roundMoney(formulaTotal)} USD.`;

  return {
    ...partial,
    estimated_value_low: range.low,
    estimated_value_high: range.high,
    estimated_stone_total: stoneTotal,
    estimated_setting_total: settingTotal,
    inferred_complexity_multiplier: complexityMultiplier,
    estimated_formula_total: formulaTotal,
    pricing_summary: canonicalPricingSummary,
    reasoning:
      reasoning ||
      "Gemini output was normalized against catalog anchors, metal rates, inferred weight, and the internal complexity grid before the final formula was applied.",
    recommended_next_step:
      recommendedNextStep || "Review the inferred weight and nearest catalog match before sending the quote.",
    matched_catalog_stone_id: descriptionHasStone
      ? normalizeTextValue(partial.matched_catalog_stone_id, matchedStone?.stone_id ?? "")
      : "",
    matched_catalog_setting_id: normalizeTextValue(partial.matched_catalog_setting_id, matchedSetting?.setting_id ?? ""),
    inferred_metal: inferredMetal,
    inferred_carat: inferredCarat,
    inferred_complexity_level: inferredComplexity,
    inferred_gold_weight_g: inferredWeight,
  };
}

function normalizeValuationEstimatePayload(payload: unknown): unknown {
  if (!payload || typeof payload !== "object") {
    return payload;
  }

  const source = payload as Record<string, unknown>;

  return {
    ...source,
    inferred_valuation_target: normalizeValuationTarget(source.inferred_valuation_target),
    estimated_value_low: normalizeNumericValue(source.estimated_value_low),
    estimated_value_high: normalizeNumericValue(source.estimated_value_high),
    pricing_summary: normalizeTextValue(source.pricing_summary, "No pricing summary logged."),
    reasoning: normalizeTextValue(source.reasoning, "Estimated from the description, catalog context, and grounded pricing cues."),
    recommended_next_step: normalizeTextValue(
      source.recommended_next_step,
      "Review the inferred match and adjust the quote if the piece differs materially.",
    ),
    estimated_stone_total: normalizeNumericValue(source.estimated_stone_total),
    estimated_setting_total: normalizeNumericValue(source.estimated_setting_total),
    inferred_complexity_multiplier: normalizeNumericValue(source.inferred_complexity_multiplier),
    estimated_formula_total: normalizeNumericValue(source.estimated_formula_total),
    matched_catalog_stone_id: normalizeTextValue(source.matched_catalog_stone_id),
    matched_catalog_setting_id: normalizeTextValue(source.matched_catalog_setting_id),
    inferred_stone_type: normalizeTextValue(source.inferred_stone_type),
    inferred_stone_shape: normalizeTextValue(source.inferred_stone_shape),
    inferred_stone_cut: normalizeTextValue(source.inferred_stone_cut),
    inferred_setting_style: normalizeTextValue(source.inferred_setting_style),
    inferred_metal: normalizeTextValue(source.inferred_metal),
    inferred_carat: normalizeNumericValue(source.inferred_carat),
    inferred_complexity_level: normalizeNumericValue(source.inferred_complexity_level),
    inferred_gold_weight_g: normalizeNumericValue(source.inferred_gold_weight_g),
    grounding_search_queries: Array.isArray(source.grounding_search_queries)
      ? source.grounding_search_queries.map((value) => String(value).trim()).filter(Boolean)
      : [],
    grounding_sources: Array.isArray(source.grounding_sources)
      ? source.grounding_sources
          .map((value) => {
            if (!value || typeof value !== "object") {
              return null;
            }

            const sourceValue = value as Record<string, unknown>;
            const title = String(sourceValue.title ?? "").trim();
            const uri = String(sourceValue.uri ?? "").trim();

            return title && uri ? { title, uri } : null;
          })
          .filter((value): value is { title: string; uri: string } => value !== null)
      : [],
  };
}

function buildConversationTranscript(history: ValuationMessage[]) {
  if (!history.length) {
    return "No follow-up conversation yet.";
  }

  return history
    .map((message) => `${message.role.toUpperCase()}: ${message.content}`)
    .join("\n\n");
}

function matchesDescription(haystack: string, description: string) {
  const normalizedHaystack = haystack.toLowerCase();
  const tokens = tokenizeDescription(description).slice(0, 24);

  if (!tokens.length) {
    return true;
  }

  return tokens.some((token) => normalizedHaystack.includes(token));
}

function takeMatchingOrFallback<T>(items: T[], matcher: (item: T) => boolean, limit: number) {
  const matched = items.filter(matcher).slice(0, limit);
  return matched.length ? matched : items.slice(0, limit);
}

export class GeminiValuationProvider implements ValuationProvider {
  providerName = "gemini" as const;

  constructor(
    private readonly apiKey: string,
    private readonly model: string,
  ) {}

  async estimate(
    input: ValuationRequestInput,
    context: ValuationCatalogContext,
    options?: { history?: ValuationMessage[] },
  ): Promise<ValuationEstimate> {
    const metalRates = context.defaults.metalPrices;
    const descriptionHasStone = hasStoneIntent(input.description);

    const stoneCatalogExcerpt = (descriptionHasStone
      ? takeMatchingOrFallback(
          context.stones,
          (stone) => matchesDescription(`${stone.name} ${stone.shape} ${stone.color} ${stone.quality}`, input.description),
          8,
        )
      : []
    )
      .map((stone) => ({
        stone_id: stone.stone_id,
        name: stone.name,
        shape: stone.shape,
        color: stone.color,
        quality: stone.quality,
        carat: stone.carat,
        final_price: stone.final_price,
      }));

    const settingCatalogExcerpt = takeMatchingOrFallback(
      context.settings,
      (setting) => matchesDescription(`${setting.style} ${setting.metal} ${setting.stone_capacity}`, input.description),
      8,
    )
      .map((setting) => ({
        setting_id: setting.setting_id,
        style: setting.style,
        metal: setting.metal,
        complexity_level: setting.complexity_level,
        gold_weight_g: setting.gold_weight_g,
        labor_cost: setting.labor_cost,
        setting_price_14k: setting.base_price,
      }));

    const pricingMethod = [
      "Reason through the estimate in this exact order before producing the final JSON:",
      "0. Treat the description as a full design brief from an internal jewelry employee. Read it the way an experienced jeweler or estimator would, using the whole description before deciding what the piece most likely is.",
      "0a. Infer the likely target, stone details, metal, setting complexity, weight, and construction cues from the whole description when those fields are not explicitly provided.",
      "1. Find the closest catalog stone and setting matches first when they exist.",
      "2. Use Google Search grounding to verify live pricing context when it materially affects the estimate, especially gold prices, other precious metal pricing, stone prices, and comparable jewelry market anchors.",
      "3. Return estimated_stone_total as the best stone subtotal in USD before any complexity multiplier.",
      "4. Return estimated_setting_total as the best setting subtotal in USD before any complexity multiplier, using setting price, gold weight, metal rate, labor, and grounded comparables when needed.",
      "5. Infer complexity level from 1 to 5 when the user did not state it. Use the internal grid: 1 very simple, 2 moderate, 3 standard bespoke, 4 detailed, 5 highly intricate.",
      "6. Use the provided metal rates as the first numeric anchor. If grounded search finds materially newer market data, mention that in pricing_summary and reasoning, but keep the final estimate practical for internal quoting.",
      "7. The application will compute the final formula total as (estimated_stone_total + estimated_setting_total) * complexity multiplier. Your JSON must provide the stone subtotal, setting subtotal, and inferred complexity cleanly.",
      "8. Build a tight low/high range around that same formula basis if you return a range at all. Do not return a range that contradicts the formula basis.",
      "8. Do not describe the model as learning from employee behavior. Requests are logged only for later prompt/process improvement.",
      "9. Use all relevant clues across the full description together. Do not anchor only on the first keyword or first noun phrase if later details add important context.",
      "10. When the description is ambiguous, make the most commercially sensible internal quoting assumption and say so briefly in the reasoning.",
    ].join("\n");

    const prompt = [
      "Estimate a practical internal jewelry value range for sourcing and quoting, not a public retail appraisal.",
      "Think like a senior jewelry estimator receiving a natural-language brief from a colleague.",
      "The description may be messy, incomplete, informal, or written in business shorthand. Your job is to interpret it contextually and turn it into a useful internal approximation.",
      "Return strict JSON with the keys: estimated_value_low, estimated_value_high, estimated_stone_total, estimated_setting_total, inferred_complexity_multiplier, estimated_formula_total, pricing_summary, reasoning, recommended_next_step, matched_catalog_stone_id, matched_catalog_setting_id, inferred_valuation_target, inferred_stone_type, inferred_stone_shape, inferred_stone_cut, inferred_setting_style, inferred_metal, inferred_carat, inferred_complexity_level, inferred_gold_weight_g, grounding_search_queries, grounding_sources.",
      'Example numeric style: {"estimated_stone_total": 320, "estimated_setting_total": 450, "inferred_complexity_level": 3, "estimated_formula_total": 2156, "estimated_value_low": 2156, "estimated_value_high": 2156}.',
      "If no catalog match exists, set the matched catalog field to an empty string.",
      "Use the inferred_* fields to return the structured characteristics you extracted from the description.",
      "If a characteristic cannot be inferred, return an empty string for text fields and 0 for numeric fields.",
      "All numeric fields must be raw JSON numbers only. Do not include units, currency symbols, words, or formatted strings in numeric fields.",
      "Do not use null, NaN, unknown, or explanatory text in numeric fields. Always output concrete numbers for estimated_stone_total, estimated_setting_total, and inferred_complexity_level.",
      "If no stone is described or implied, estimated_stone_total must be 0.",
      "When a close setting style exists in the provided catalog excerpt, prefer that setting's complexity level as the anchor. Simple signet rings should resolve to complexity level 1 unless the brief clearly describes extra intricate work.",
      "If the description gives weight or material but not a catalog match, still produce an estimate from the metal rates and a practical making/labor assumption.",
      "pricing_summary must be a concise numeric pricing trace, not hidden chain-of-thought. Keep it to 3-5 short sentences with the main amounts and basis used, including stone subtotal, setting subtotal, complexity, multiplier, and formula total.",
      "reasoning must stay short.",
      "recommended_next_step must stay short.",
      "grounding_search_queries should list the main web-search queries you actually used, if any.",
      "grounding_sources should be an array of objects with title and uri for the key web sources you relied on. If grounding was not needed, return an empty array.",
      pricingMethod,
      "",
      `Request description: ${input.description}`,
      `Reference image URL: ${input.reference_image_url || "Not provided"}`,
      input.reference_image_url
        ? "A reference URL was provided. Use it as supplemental context if it helps identify the piece or comparable listing."
        : "No reference URL was provided.",
      `Conversation so far:\n${buildConversationTranscript(options?.history ?? [])}`,
      "Infer the actual jewelry characteristics from the full description and catalog context.",
      "The description may be long and detailed. Use the entire description holistically, including metal references, weights, setting construction, stone arrangement, dimensions, finish, inspiration, style cues, era references, and any pricing clues implied by the brief.",
      `Provided metal rates per gram: ${JSON.stringify(metalRates)}`,
      "Use the inferred characteristics, provided metal rates, and catalog values as the primary basis of the estimate. Do not ignore weight, labor, or catalog final_price/setting price values.",
      `Known catalog stones: ${JSON.stringify(stoneCatalogExcerpt)}`,
      `Known catalog settings: ${JSON.stringify(settingCatalogExcerpt)}`,
    ].join("\n");

    const parts: Array<Record<string, unknown>> = [{ text: prompt }];
    const parsedImage = input.image_data_url ? parseDataUrl(input.image_data_url) : null;

    if (parsedImage) {
      parts.push({
        inlineData: {
          mimeType: parsedImage.mimeType,
          data: parsedImage.data,
        },
      });
    }

    const response = await runWithGeminiModelFallback(this.apiKey, this.model, (ai, model) =>
      ai.models.generateContent({
        model,
        contents: [
          {
            role: "user",
            parts,
          },
        ],
        config: {
          tools: [{ googleSearch: {} }],
          ...(model.includes("flash") ? {} : { responseMimeType: "application/json" }),
          temperature: 0.1,
          systemInstruction:
            "You are Capucinne's internal jewelry valuation assistant. Use grounded search when current market data helps, especially for gold, precious metals, gemstones, and comparable jewelry pricing. Return only JSON.",
        },
      }),
    );

    const text = String((response as { text?: string }).text ?? "").trim();

    if (!text) {
      throw new Error("Gemini returned an empty valuation response.");
    }

    const normalizedPayload = normalizeValuationEstimatePayload(JSON.parse(extractJsonText(text)));
    const parsed = valuationEstimateSchema.parse(
      normalizedPayload,
    );
    const grounding = extractGrounding(
      response as {
        text?: string;
        candidates?: Array<{
          groundingMetadata?: {
            webSearchQueries?: string[];
            groundingChunks?: Array<{
              web?: {
                uri?: string;
                title?: string;
              };
            }>;
          };
        }>;
      },
    );
    const repaired = estimateFromContext(
      input,
      context,
      parsed,
      fallbackMatchedStone(context, input.description, normalizedPayload),
      fallbackMatchedSetting(context, input.description, normalizedPayload),
    );

    return {
      ...repaired,
      grounding_search_queries:
        repaired.grounding_search_queries.length > 0 ? repaired.grounding_search_queries : grounding.searchQueries,
      grounding_sources: repaired.grounding_sources.length > 0 ? repaired.grounding_sources : grounding.sources,
    };
  }
}
