import { NextRequest, NextResponse } from "next/server";
import { ZodError } from "zod";

import { estimateValuation } from "@/lib/ai";
import { getAppConfig } from "@/lib/config";
import { getRepository } from "@/lib/repositories";
import { type ValuationMessage, type ValuationRecord } from "@/lib/types";
import { createRecordId, toIsoNow } from "@/lib/utils";
import { valuationFollowUpSchema } from "@/lib/validators";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function badRequest(error: unknown) {
  if (error instanceof ZodError) {
    return NextResponse.json({ error: error.issues[0]?.message ?? "Invalid valuation follow-up payload." }, { status: 400 });
  }

  const message = error instanceof Error ? error.message : "Unexpected valuation follow-up API error.";
  return NextResponse.json({ error: message }, { status: 500 });
}

function buildAssistantMessageContent(record: Pick<ValuationRecord, "pricing_summary" | "reasoning" | "recommended_next_step">) {
  return [record.pricing_summary, record.reasoning, record.recommended_next_step].filter(Boolean).join("\n\n");
}

export async function POST(
  request: NextRequest,
  context: {
    params: Promise<{
      valuationId: string;
    }>;
  },
) {
  try {
    const repository = getRepository();
    const config = getAppConfig();
    const { valuationId } = await context.params;
    const valuation = await repository.findValuationById(valuationId);

    if (!valuation) {
      return NextResponse.json({ error: "Approximation not found." }, { status: 404 });
    }

    const input = valuationFollowUpSchema.parse(await request.json());
    const [stones, settings, pricingDefaults] = await Promise.all([
      repository.listStones(),
      repository.listSettings(),
      repository.getPricingDefaults(),
    ]);

    const createdAt = toIsoNow();
    const userMessage: ValuationMessage = {
      message_id: createRecordId("valuation_msg"),
      role: "user",
      content: input.message,
      created_at: createdAt,
    };

    const history = [...valuation.messages, userMessage];
    const { estimate, provider } = await estimateValuation(
      {
        description: valuation.description,
        reference_image_url: valuation.reference_image_url,
        image_data_url: valuation.image_data_url,
        created_by: input.created_by,
      },
      {
        stones,
        settings,
        defaults: {
          goldPricePerGram: config.goldPricePerGram,
          quoteMarginMultiplier: config.quoteMarginMultiplier,
          metalPrices: pricingDefaults.metalPrices,
          metalPricingSource: pricingDefaults.metalPricingSource,
        },
      },
      { history },
    );

    const updated: ValuationRecord = {
      ...valuation,
      valuation_target: estimate.inferred_valuation_target,
      stone_type: estimate.inferred_stone_type,
      stone_shape: estimate.inferred_stone_shape,
      stone_cut: estimate.inferred_stone_cut,
      setting_style: estimate.inferred_setting_style,
      metal: estimate.inferred_metal,
      carat: estimate.inferred_carat,
      complexity_level: estimate.inferred_complexity_level,
      gold_weight_g: estimate.inferred_gold_weight_g,
      estimated_value_low: estimate.estimated_value_low,
      estimated_value_high: estimate.estimated_value_high,
      estimated_stone_total: estimate.estimated_stone_total,
      estimated_setting_total: estimate.estimated_setting_total,
      inferred_complexity_multiplier: estimate.inferred_complexity_multiplier,
      estimated_formula_total: estimate.estimated_formula_total,
      pricing_summary: estimate.pricing_summary,
      reasoning: estimate.reasoning,
      recommended_next_step: estimate.recommended_next_step,
      matched_catalog_stone_id: estimate.matched_catalog_stone_id,
      matched_catalog_setting_id: estimate.matched_catalog_setting_id,
      inferred_valuation_target: estimate.inferred_valuation_target,
      inferred_stone_type: estimate.inferred_stone_type,
      inferred_stone_shape: estimate.inferred_stone_shape,
      inferred_stone_cut: estimate.inferred_stone_cut,
      inferred_setting_style: estimate.inferred_setting_style,
      inferred_metal: estimate.inferred_metal,
      inferred_carat: estimate.inferred_carat,
      inferred_complexity_level: estimate.inferred_complexity_level,
      inferred_gold_weight_g: estimate.inferred_gold_weight_g,
      grounding_search_queries: estimate.grounding_search_queries ?? [],
      grounding_sources: estimate.grounding_sources ?? [],
      provider,
      updated_at: createdAt,
      messages: history,
    };

    const assistantMessage: ValuationMessage = {
      message_id: createRecordId("valuation_msg"),
      role: "assistant",
      content: buildAssistantMessageContent(updated),
      created_at: createdAt,
      estimated_value_low: updated.estimated_value_low,
      estimated_value_high: updated.estimated_value_high,
      estimated_formula_total: updated.estimated_formula_total,
      pricing_summary: updated.pricing_summary,
      reasoning: updated.reasoning,
      recommended_next_step: updated.recommended_next_step,
    };

    updated.messages = [...history, assistantMessage];

    const saved = await repository.updateValuation(updated);
    return NextResponse.json(saved);
  } catch (error) {
    return badRequest(error);
  }
}
