"use client";

import { type FormEvent, useDeferredValue, useEffect, useState, useTransition } from "react";

import {
  CatalogPanel,
  DetailItem,
  Field,
  InlineButton,
  pageSizes,
  QuoteItem,
  SectionCard,
  SelectionCard,
  StatusPill,
} from "@/components/dashboard-ui";
import { calculateQuoteBreakdown } from "@/lib/services/quote-service";
import {
  type DashboardSnapshot,
  type Inquiry,
  type PaginatedResult,
  type ProductComposition,
  type ProductCompositionVariant,
  type Setting,
  type Stone,
  type ValuationMessage,
  type ValuationRecord,
} from "@/lib/types";
import { formatStoneSize } from "@/lib/utils";

type ProjectForm = {
  customer_name: string;
  stone_id: string;
  setting_id: string;
  target_size_mm: string;
  target_gold_weight_g: string;
  status: string;
  reference_image_url: string;
  created_by: string;
};

type ValuationForm = {
  description: string;
  reference_image_url: string;
  image_data_url: string;
};

type ProjectStoneLine = {
  stone: Stone;
  quantity: number;
};

type ToastItem = {
  id: string;
  message: string;
  tone: "success" | "error";
};

type RecallVariantSummary = {
  missingStoneCount: number;
  missingSettingCount: number;
};

type StoneBrowseFilters = {
  stoneId: string;
  name: string;
  shape: string;
  color: string;
  quality: string;
  size: string;
  minCarat: string;
  maxCarat: string;
  minPricePerCarat: string;
  maxPricePerCarat: string;
};

type SettingBrowseFilters = {
  productId: string;
  settingId: string;
  style: string;
  metal: string;
  minWeightG: string;
  maxWeightG: string;
  minComplexity: string;
  maxComplexity: string;
  minLaborCost: string;
  maxLaborCost: string;
  minBasePrice: string;
  maxBasePrice: string;
};

type StoneAssistFilters = Partial<{
  stoneId: string;
  name: string;
  shape: string;
  color: string;
  quality: string;
  size: string;
  minCarat: number;
  maxCarat: number;
  minPricePerCarat: number;
  maxPricePerCarat: number;
}>;

type SettingAssistFilters = Partial<{
  settingId: string;
  style: string;
  metal: string;
  minWeightG: number;
  maxWeightG: number;
  minComplexity: number;
  maxComplexity: number;
  minLaborCost: number;
  maxLaborCost: number;
  minBasePrice: number;
  maxBasePrice: number;
}>;

type CatalogAssistResponse<TFilters> = {
  provider: "gemini";
  normalizedQuery: string;
  summary: string;
  filters: TFilters;
};

const blankProject: ProjectForm = {
  customer_name: "",
  stone_id: "",
  setting_id: "",
  target_size_mm: "",
  target_gold_weight_g: "",
  status: "open",
  reference_image_url: "",
  created_by: "atelier-team",
};

const blankValuation: ValuationForm = {
  description: "",
  reference_image_url: "",
  image_data_url: "",
};

const blankStoneBrowseFilters: StoneBrowseFilters = {
  stoneId: "",
  name: "",
  shape: "",
  color: "",
  quality: "",
  size: "",
  minCarat: "",
  maxCarat: "",
  minPricePerCarat: "",
  maxPricePerCarat: "",
};

const blankSettingBrowseFilters: SettingBrowseFilters = {
  productId: "",
  settingId: "",
  style: "",
  metal: "",
  minWeightG: "",
  maxWeightG: "",
  minComplexity: "",
  maxComplexity: "",
  minLaborCost: "",
  maxLaborCost: "",
  minBasePrice: "",
  maxBasePrice: "",
};

function money(value: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value || 0);
}

function midpoint(low: number, high: number) {
  return Math.round(((low + high) / 2) * 100) / 100;
}

function stamp(value: string) {
  if (!value) return "Imported baseline";
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function optionalNumber(value: string) {
  if (!value.trim()) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function pluralize(count: number, singular: string, plural = `${singular}s`) {
  return count === 1 ? singular : plural;
}

function excerpt(value: string, max = 88) {
  const normalized = value.trim();
  if (!normalized) return "No description";
  return normalized.length > max ? `${normalized.slice(0, max - 1)}…` : normalized;
}

function MediaPreview({ src, alt }: { src: string; alt: string }) {
  const [hidden, setHidden] = useState(false);

  if (!src.trim() || hidden) {
    return null;
  }

  return (
    <figure className="valuation-preview">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} onError={() => setHidden(true)} />
    </figure>
  );
}

function parsePositiveQuantity(value: string): number | null {
  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return null;
  }

  const normalized = Math.floor(parsed);
  return normalized >= 1 ? normalized : null;
}

function serializeStoneIds(lines: ProjectStoneLine[]) {
  return lines
    .flatMap((line) => Array.from({ length: line.quantity }, () => line.stone.stone_id))
    .join(", ");
}

function summarizeSelection(ids: string[]) {
  if (!ids.length) return "";
  if (ids.length === 1) return ids[0];
  if (ids.length === 2) return ids.join(", ");
  return `${ids.slice(0, 2).join(", ")} +${ids.length - 2} more`;
}

function buildAssistantMessagePreview(message: ValuationMessage) {
  return message.pricing_summary || message.content;
}

function summarizeRecallVariant(variant: ProductCompositionVariant): RecallVariantSummary {
  return {
    missingStoneCount: variant.stones.filter((line) => !line.stone).length,
    missingSettingCount: variant.setting_ids.filter(
      (settingId) => !variant.settings.some((setting) => setting.setting_id === settingId),
    ).length,
  };
}

function countFilledFilters(filters: Record<string, string>) {
  return Object.values(filters).filter((value) => value.trim().length > 0).length;
}

function countAppliedAssistFilters(filters: Record<string, string | number | undefined>) {
  return Object.values(filters).filter((value) => value !== undefined && value !== "").length;
}

function stringifyFilterValue(value: string | number | undefined) {
  return value === undefined || value === "" ? "" : String(value);
}

async function postJson<T>(url: string, body: Record<string, unknown>) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = (await response.json()) as T & { error?: string };
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

async function getJson<T>(url: string, signal?: AbortSignal) {
  const response = await fetch(url, { cache: "no-store", signal });
  const payload = (await response.json()) as T & { error?: string };
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

function toDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

async function fetchCatalogPage<T>(
  resource: "stones" | "settings",
  params: { query: string; page: number; pageSize: number; filters?: Record<string, string> },
  signal?: AbortSignal,
) {
  const searchParams = new URLSearchParams({ page: String(params.page), pageSize: String(params.pageSize) });
  if (params.query.trim()) searchParams.set("query", params.query.trim());
  Object.entries(params.filters ?? {}).forEach(([key, value]) => {
    if (value.trim()) {
      searchParams.set(key, value.trim());
    }
  });
  return getJson<PaginatedResult<T>>(`/api/${resource}?${searchParams.toString()}`, signal);
}

export function DashboardApp({ initialSnapshotJson }: { initialSnapshotJson: string }) {
  const [initialSnapshot] = useState<DashboardSnapshot>(() => JSON.parse(initialSnapshotJson) as DashboardSnapshot);
  const [stones, setStones] = useState(initialSnapshot.stones);
  const [settings, setSettings] = useState(initialSnapshot.settings);
  const [projects, setProjects] = useState(initialSnapshot.inquiries);
  const [valuations, setValuations] = useState(initialSnapshot.valuations);
  const [projectForm, setProjectForm] = useState<ProjectForm>(blankProject);
  const [valuationForm, setValuationForm] = useState(blankValuation);
  const [stoneBrowseFilters, setStoneBrowseFilters] = useState<StoneBrowseFilters>(blankStoneBrowseFilters);
  const [settingBrowseFilters, setSettingBrowseFilters] = useState<SettingBrowseFilters>(blankSettingBrowseFilters);
  const [stoneSearch, setStoneSearch] = useState("");
  const [settingSearch, setSettingSearch] = useState("");
  const [stonePage, setStonePage] = useState(1);
  const [settingPage, setSettingPage] = useState(1);
  const [stonePageSize, setStonePageSize] = useState<(typeof pageSizes)[number]>(20);
  const [settingPageSize, setSettingPageSize] = useState<(typeof pageSizes)[number]>(20);
  const [stoneTotal, setStoneTotal] = useState(initialSnapshot.kpis.stoneCount);
  const [settingTotal, setSettingTotal] = useState(initialSnapshot.kpis.settingCount);
  const [stoneTotalPages, setStoneTotalPages] = useState(Math.max(1, Math.ceil(initialSnapshot.kpis.stoneCount / 20)));
  const [settingTotalPages, setSettingTotalPages] = useState(Math.max(1, Math.ceil(initialSnapshot.kpis.settingCount / 20)));
  const [selectedStone, setSelectedStone] = useState<Stone | null>(initialSnapshot.stones[0] ?? null);
  const [selectedSetting, setSelectedSetting] = useState<Setting | null>(initialSnapshot.settings[0] ?? null);
  const [selectedStoneQuantity, setSelectedStoneQuantity] = useState("1");
  const [projectStoneRefs, setProjectStoneRefs] = useState<ProjectStoneLine[]>([]);
  const [projectSettingRefs, setProjectSettingRefs] = useState<Setting[]>([]);
  const [recallReference, setRecallReference] = useState("");
  const [recallLoading, setRecallLoading] = useState(false);
  const [recallNotice, setRecallNotice] = useState("");
  const [recalledProduct, setRecalledProduct] = useState<ProductComposition | null>(null);
  const [recalledVariantKey, setRecalledVariantKey] = useState("");
  const [projectNotice, setProjectNotice] = useState("");
  const [valuationNotice, setValuationNotice] = useState("");
  const [valuationFollowUp, setValuationFollowUp] = useState("");
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [stoneError, setStoneError] = useState("");
  const [settingError, setSettingError] = useState("");
  const [stoneLoading, setStoneLoading] = useState(false);
  const [settingLoading, setSettingLoading] = useState(false);
  const [stoneAssistLoading, setStoneAssistLoading] = useState(false);
  const [settingAssistLoading, setSettingAssistLoading] = useState(false);
  const [stoneAssistNotice, setStoneAssistNotice] = useState("");
  const [settingAssistNotice, setSettingAssistNotice] = useState("");
  const [valuationResult, setValuationResult] = useState<ValuationRecord | null>(initialSnapshot.valuations[0] ?? null);
  const [valuationModal, setValuationModal] = useState<ValuationRecord | null>(null);
  const [valuationLoading, setValuationLoading] = useState(false);
  const [valuationFollowUpLoading, setValuationFollowUpLoading] = useState(false);
  const [isPending, startTransition] = useTransition();

  const deferredStoneSearch = useDeferredValue(stoneSearch.trim());
  const deferredSettingSearch = useDeferredValue(settingSearch.trim());
  const stoneSuggestions = stoneSearch.trim() ? stones.slice(0, 8) : [];
  const settingSuggestions = settingSearch.trim() ? settings.slice(0, 8) : [];

  function updateStoneBrowseFilter<Key extends keyof StoneBrowseFilters>(key: Key, value: StoneBrowseFilters[Key]) {
    setStoneBrowseFilters((current) => ({ ...current, [key]: value }));
    setStonePage(1);
  }

  function updateSettingBrowseFilter<Key extends keyof SettingBrowseFilters>(key: Key, value: SettingBrowseFilters[Key]) {
    setSettingBrowseFilters((current) => ({ ...current, [key]: value }));
    setSettingPage(1);
  }

  function applyStoneAssistFilters(filters: StoneAssistFilters) {
    setStoneBrowseFilters({
      ...blankStoneBrowseFilters,
      stoneId: stringifyFilterValue(filters.stoneId),
      name: stringifyFilterValue(filters.name),
      shape: stringifyFilterValue(filters.shape),
      color: stringifyFilterValue(filters.color),
      quality: stringifyFilterValue(filters.quality),
      size: stringifyFilterValue(filters.size),
      minCarat: stringifyFilterValue(filters.minCarat),
      maxCarat: stringifyFilterValue(filters.maxCarat),
      minPricePerCarat: stringifyFilterValue(filters.minPricePerCarat),
      maxPricePerCarat: stringifyFilterValue(filters.maxPricePerCarat),
    });
    setStonePage(1);
  }

  function applySettingAssistFilters(filters: SettingAssistFilters) {
    setSettingBrowseFilters({
      ...blankSettingBrowseFilters,
      settingId: stringifyFilterValue(filters.settingId),
      style: stringifyFilterValue(filters.style),
      metal: stringifyFilterValue(filters.metal),
      minWeightG: stringifyFilterValue(filters.minWeightG),
      maxWeightG: stringifyFilterValue(filters.maxWeightG),
      minComplexity: stringifyFilterValue(filters.minComplexity),
      maxComplexity: stringifyFilterValue(filters.maxComplexity),
      minLaborCost: stringifyFilterValue(filters.minLaborCost),
      maxLaborCost: stringifyFilterValue(filters.maxLaborCost),
      minBasePrice: stringifyFilterValue(filters.minBasePrice),
      maxBasePrice: stringifyFilterValue(filters.maxBasePrice),
    });
    setSettingPage(1);
  }

  function pushToast(message: string, tone: ToastItem["tone"] = "success") {
    const id = crypto.randomUUID();
    setToasts((current) => [...current, { id, message, tone }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 3200);
  }

  async function handleCatalogAssist(target: "stone" | "setting") {
    const query = (target === "stone" ? stoneSearch : settingSearch).trim();

    if (!query) {
      pushToast(`Type a ${target} description first, then use AI filter.`, "error");
      return;
    }

    if (target === "stone") {
      setStoneAssistLoading(true);
    } else {
      setSettingAssistLoading(true);
    }

    try {
      if (target === "stone") {
        const payload = await postJson<CatalogAssistResponse<StoneAssistFilters>>("/api/catalog-assist", { target, query });
        applyStoneAssistFilters(payload.filters);
        setStoneSearch(payload.normalizedQuery);
        setStoneAssistNotice(`${payload.provider === "gemini" ? "Gemini" : "Search assist"}: ${payload.summary}`);
        pushToast(
          `${payload.provider === "gemini" ? "Gemini" : "Search assist"} applied ${countAppliedAssistFilters(payload.filters)} stone filter${countAppliedAssistFilters(payload.filters) === 1 ? "" : "s"}.`,
        );
      } else {
        const payload = await postJson<CatalogAssistResponse<SettingAssistFilters>>("/api/catalog-assist", { target, query });
        applySettingAssistFilters(payload.filters);
        setSettingSearch(payload.normalizedQuery);
        setSettingAssistNotice(`${payload.provider === "gemini" ? "Gemini" : "Search assist"}: ${payload.summary}`);
        pushToast(
          `${payload.provider === "gemini" ? "Gemini" : "Search assist"} applied ${countAppliedAssistFilters(payload.filters)} setting filter${countAppliedAssistFilters(payload.filters) === 1 ? "" : "s"}.`,
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not interpret the search description.";
      if (target === "stone") {
        setStoneAssistNotice(message);
      } else {
        setSettingAssistNotice(message);
      }
      pushToast(message, "error");
    } finally {
      if (target === "stone") {
        setStoneAssistLoading(false);
      } else {
        setSettingAssistLoading(false);
      }
    }
  }

  useEffect(() => {
    setSelectedStoneQuantity("1");
  }, [selectedStone?.stone_id]);

  useEffect(() => {
    if (!valuationModal) {
      return;
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setValuationModal(null);
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [valuationModal]);

  useEffect(() => {
    const controller = new AbortController();
    setStoneLoading(true);
    setStoneError("");
    fetchCatalogPage<Stone>(
      "stones",
      {
        query: deferredStoneSearch,
        page: stonePage,
        pageSize: stonePageSize,
        filters: stoneBrowseFilters,
      },
      controller.signal,
    )
      .then((payload) => {
        setStones(payload.items);
        setStoneTotal(payload.total);
        setStoneTotalPages(payload.totalPages);
        setSelectedStone((current) => payload.items.find((stone) => stone.stone_id === current?.stone_id) ?? payload.items[0] ?? null);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setStoneError(error instanceof Error ? error.message : "Could not load stones.");
        setStones([]);
        setStoneTotal(0);
        setStoneTotalPages(1);
        setSelectedStone(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setStoneLoading(false);
      });
    return () => controller.abort();
  }, [deferredStoneSearch, stonePage, stonePageSize, stoneBrowseFilters]);

  useEffect(() => {
    const controller = new AbortController();
    setSettingLoading(true);
    setSettingError("");
    fetchCatalogPage<Setting>(
      "settings",
      {
        query: deferredSettingSearch,
        page: settingPage,
        pageSize: settingPageSize,
        filters: settingBrowseFilters,
      },
      controller.signal,
    )
      .then((payload) => {
        setSettings(payload.items);
        setSettingTotal(payload.total);
        setSettingTotalPages(payload.totalPages);
        setSelectedSetting((current) => payload.items.find((setting) => setting.setting_id === current?.setting_id) ?? payload.items[0] ?? null);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setSettingError(error instanceof Error ? error.message : "Could not load settings.");
        setSettings([]);
        setSettingTotal(0);
        setSettingTotalPages(1);
        setSelectedSetting(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setSettingLoading(false);
      });
    return () => controller.abort();
  }, [deferredSettingSearch, settingPage, settingPageSize, settingBrowseFilters]);

  const expandedProjectStones = projectStoneRefs.flatMap((line) => Array.from({ length: line.quantity }, () => line.stone));
  const totalProjectStoneQuantity = projectStoneRefs.reduce((sum, line) => sum + line.quantity, 0);

  const projectQuote = calculateQuoteBreakdown({
    stones: expandedProjectStones,
    settings: projectSettingRefs,
    settingReferenceText: projectForm.setting_id,
    targetGoldWeightG: optionalNumber(projectForm.target_gold_weight_g),
    goldPricePerGram: initialSnapshot.defaults.goldPricePerGram,
    metalPrices: initialSnapshot.defaults.metalPrices,
    quoteMarginMultiplier: initialSnapshot.defaults.quoteMarginMultiplier,
  });

  const projectStoneIds = projectStoneRefs.map((line) => (line.quantity > 1 ? `${line.stone.stone_id} x${line.quantity}` : line.stone.stone_id));
  const projectSettingIds = projectSettingRefs.map((setting) => setting.setting_id);
  const stoneActiveFilterCount = countFilledFilters(stoneBrowseFilters);
  const settingActiveFilterCount = countFilledFilters(settingBrowseFilters);
  const activeRecalledVariant =
    recalledProduct?.variants.find((variant) => variant.variant_key === recalledVariantKey) ??
    recalledProduct?.variants[0] ??
    null;
  const valuationThread = valuationResult?.messages ?? [];
  const valuationFollowUpMessages = valuationThread.slice(2);

  function attachStoneToProject(stone: Stone) {
    const quantity = parsePositiveQuantity(selectedStoneQuantity);

    if (!quantity) {
      pushToast("Enter a stone quantity of 1 or more before adding it to the listing.", "error");
      return;
    }

    const existing = projectStoneRefs.find((line) => line.stone.stone_id === stone.stone_id);
    const next = existing
      ? projectStoneRefs.map((line) =>
          line.stone.stone_id === stone.stone_id ? { ...line, quantity: line.quantity + quantity } : line,
        )
      : [...projectStoneRefs, { stone, quantity }];

    setProjectStoneRefs(next);
    setProjectForm((form) => ({ ...form, stone_id: serializeStoneIds(next) }));
    setSelectedStoneQuantity("1");
    pushToast(
      existing
        ? `Stone quantity updated. ${stone.name} is now at ${existing.quantity + quantity}.`
        : `Added ${quantity}x ${stone.name} to the listing.`,
    );
  }

  function attachSettingToProject(setting: Setting) {
    if (projectSettingRefs.some((item) => item.setting_id === setting.setting_id)) {
      pushToast("That catalog setting is already attached to the listing.", "error");
      return;
    }

    const next = [...projectSettingRefs, setting];
    setProjectSettingRefs(next);
    setProjectForm((form) => ({
      ...form,
      setting_id: next.map((item) => item.setting_id).join(", "),
    }));
    pushToast(`${setting.style} added to the listing.`);
  }

  function removeStoneFromProject(stoneId: string) {
    const next = projectStoneRefs.filter((line) => line.stone.stone_id !== stoneId);
    setProjectStoneRefs(next);
    setProjectForm((form) => ({ ...form, stone_id: serializeStoneIds(next) }));
  }

  function removeSettingFromProject(settingId: string) {
    const next = projectSettingRefs.filter((setting) => setting.setting_id !== settingId);
    setProjectSettingRefs(next);
    setProjectForm((form) => ({ ...form, setting_id: next.map((setting) => setting.setting_id).join(", ") }));
  }

  function upsertValuationRecord(next: ValuationRecord) {
    setValuations((current) => {
      const existingIndex = current.findIndex((valuation) => valuation.valuation_id === next.valuation_id);

      if (existingIndex === -1) {
        return [next, ...current];
      }

      const updated = [...current];
      updated[existingIndex] = next;
      return updated.sort((left, right) => right.updated_at.localeCompare(left.updated_at));
    });
    setValuationResult(next);
    setValuationModal((current) => (current?.valuation_id === next.valuation_id ? next : current));
  }

  function applyRecalledVariant(composition: ProductComposition, variantKey: string) {
    const variant =
      composition.variants.find((candidate) => candidate.variant_key === variantKey) ??
      composition.variants.find((candidate) => candidate.variant_key === composition.default_variant_key) ??
      composition.variants[0];

    if (!variant) {
      setRecallNotice("That Shopify listing did not return any usable variant rows.");
      pushToast("No usable variant rows were found for that listing.", "error");
      return;
    }

    const nextProjectStoneRefs = variant.stones
      .filter((line) => line.stone)
      .map((line) => ({ stone: line.stone as Stone, quantity: line.quantity }));
    const nextProjectSettingRefs = variant.settings;
    const missing = summarizeRecallVariant(variant);

    setRecalledProduct(composition);
    setRecalledVariantKey(variant.variant_key);
    setProjectStoneRefs(nextProjectStoneRefs);
    setProjectSettingRefs(nextProjectSettingRefs);
    setProjectForm((current) => ({
      ...current,
      customer_name: composition.title || current.customer_name,
      stone_id: serializeStoneIds(nextProjectStoneRefs),
      setting_id: nextProjectSettingRefs.map((setting) => setting.setting_id).join(", "),
      reference_image_url: /^https?:\/\//i.test(recallReference.trim()) ? recallReference.trim() : current.reference_image_url,
    }));
    setSelectedStone(nextProjectStoneRefs[0]?.stone ?? selectedStone);
    setSelectedSetting(nextProjectSettingRefs[0] ?? selectedSetting);

    const noteParts = [
      `Loaded ${composition.title || composition.product_handle}`,
      `${nextProjectStoneRefs.length} stone SKU`,
      `${nextProjectSettingRefs.length} setting SKU`,
      missing.missingStoneCount > 0 ? `${missing.missingStoneCount} stone SKU still missing from catalog` : "",
      missing.missingSettingCount > 0 ? `${missing.missingSettingCount} setting SKU still missing from catalog` : "",
    ].filter(Boolean);

    const message = noteParts.join(" | ");
    setRecallNotice(message);
    setProjectNotice(`Recalled ${composition.product_id} into the listing builder.`);
    pushToast(message);
  }

  function loadValuationIntoForm(valuation: ValuationRecord) {
    setValuationForm({
      description: valuation.description,
      reference_image_url: valuation.reference_image_url,
      image_data_url: valuation.image_data_url,
    });
    setValuationResult(valuation);
    setValuationModal(null);
    setValuationFollowUp("");
    setValuationNotice(`Loaded approximation ${valuation.valuation_id} back into the form.`);
    pushToast(`Approximation ${valuation.valuation_id} loaded into the form.`);
  }

  async function handleValuationImageChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    if (!file) {
      setValuationForm((current) => ({ ...current, image_data_url: "" }));
      return;
    }

    try {
      const imageDataUrl = await toDataUrl(file);
      setValuationForm((current) => ({ ...current, image_data_url: imageDataUrl }));
      pushToast(`${file.name} attached for Gemini.`);
    } catch (error) {
      setValuationNotice(error instanceof Error ? error.message : "Could not attach image.");
      pushToast(error instanceof Error ? error.message : "Could not attach image.", "error");
    }
  }

  function openValuationDetails(valuation: ValuationRecord) {
    setValuationResult(valuation);
    setValuationModal(valuation);
  }

  async function runRecall() {
    if (!recallReference.trim()) {
      pushToast("Paste a Shopify ID or product URL first.", "error");
      return;
    }

    setRecallLoading(true);
    setRecallNotice("");

    try {
      const composition = await getJson<ProductComposition>(`/api/recall?reference=${encodeURIComponent(recallReference.trim())}`);
      applyRecalledVariant(composition, composition.default_variant_key);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not recall the Shopify listing.";
      setRecallNotice(message);
      pushToast(message, "error");
    } finally {
      setRecallLoading(false);
    }
  }

  async function handleProjectSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    try {
      const created = await postJson<Inquiry>("/api/inquiries", {
        customer_name: projectForm.customer_name,
        stone_id: projectForm.stone_id,
        setting_id: projectForm.setting_id,
        target_size_mm: projectForm.target_size_mm,
        target_gold_weight_g: projectForm.target_gold_weight_g,
        status: projectForm.status,
        reference_image_url: projectForm.reference_image_url,
        created_by: projectForm.created_by,
      });
      startTransition(() => setProjects((current) => [created, ...current]));
      setProjectNotice(
        `Custom listing saved with 14K ${money(created.estimated_quote)} and 18K ${money(created.estimated_quote_18k ?? created.estimated_quote)}.`,
      );
    } catch (error) {
      setProjectNotice(error instanceof Error ? error.message : "Could not save custom listing.");
    }
  }

  async function handleValuationSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setValuationLoading(true);
    setValuationNotice("");

    try {
      const created = await postJson<ValuationRecord>("/api/valuations", {
        description: valuationForm.description,
        reference_image_url: valuationForm.reference_image_url,
        image_data_url: valuationForm.image_data_url,
      });
      startTransition(() => {
        upsertValuationRecord(created);
      });
      setValuationForm(blankValuation);
      setValuationFollowUp("");
      setValuationNotice(
        `AI approximation logged with range ${money(created.estimated_value_low)} - ${money(created.estimated_value_high)}.`,
      );
    } catch (error) {
      setValuationNotice(error instanceof Error ? error.message : "Could not run AI approximation.");
    } finally {
      setValuationLoading(false);
    }
  }

  async function handleValuationFollowUpSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!valuationResult) {
      pushToast("Run or select an approximation first.", "error");
      return;
    }

    if (!valuationFollowUp.trim()) {
      pushToast("Type a refinement for Gemini first.", "error");
      return;
    }

    setValuationFollowUpLoading(true);
    setValuationNotice("");

    try {
      const updated = await postJson<ValuationRecord>(
        `/api/valuations/${encodeURIComponent(valuationResult.valuation_id)}/messages`,
        { message: valuationFollowUp.trim() },
      );
      startTransition(() => {
        upsertValuationRecord(updated);
      });
      setValuationFollowUp("");
      setValuationNotice(`Gemini refined approximation ${updated.valuation_id}.`);
      pushToast(`Gemini updated ${updated.valuation_id}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not refine the approximation.";
      setValuationNotice(message);
      pushToast(message, "error");
    } finally {
      setValuationFollowUpLoading(false);
    }
  }

  const catalogSections = (
    <>
      <SectionCard eyebrow="Stones" title="Stone catalog" compact>
        <CatalogPanel
          kind="stone"
          searchValue={stoneSearch}
          searchPlaceholder="Search SKU, stone, shape"
          searchActions={
            <InlineButton disabled={stoneAssistLoading || !stoneSearch.trim()} onClick={() => handleCatalogAssist("stone")}>
              {stoneAssistLoading ? "Filtering..." : "AI filter"}
            </InlineButton>
          }
          searchNote={stoneAssistNotice || undefined}
          filters={
            <>
              <div className="catalog-filter-grid">
                <Field label="SKU">
                  <input
                    className="field-control"
                    value={stoneBrowseFilters.stoneId}
                    onChange={(event) => updateStoneBrowseFilter("stoneId", event.target.value)}
                    placeholder="NSDI..."
                  />
                </Field>
                <Field label="Stone / type">
                  <input
                    className="field-control"
                    value={stoneBrowseFilters.name}
                    onChange={(event) => updateStoneBrowseFilter("name", event.target.value)}
                    placeholder="Diamond, sapphire"
                  />
                </Field>
                <Field label="Shape">
                  <input
                    className="field-control"
                    value={stoneBrowseFilters.shape}
                    onChange={(event) => updateStoneBrowseFilter("shape", event.target.value)}
                    placeholder="Round, oval"
                  />
                </Field>
                <Field label="Color">
                  <input
                    className="field-control"
                    value={stoneBrowseFilters.color}
                    onChange={(event) => updateStoneBrowseFilter("color", event.target.value)}
                    placeholder="White, blue"
                  />
                </Field>
                <Field label="Cut / quality">
                  <input
                    className="field-control"
                    value={stoneBrowseFilters.quality}
                    onChange={(event) => updateStoneBrowseFilter("quality", event.target.value)}
                    placeholder="Brilliant, VS"
                  />
                </Field>
                <Field label="Size">
                  <input
                    className="field-control"
                    value={stoneBrowseFilters.size}
                    onChange={(event) => updateStoneBrowseFilter("size", event.target.value)}
                    placeholder="4 mm, 4x2 mm, 4 x 2, 4,0x2,0"
                  />
                </Field>
                <Field label="Min carat">
                  <input
                    className="field-control"
                    type="number"
                    step="0.001"
                    value={stoneBrowseFilters.minCarat}
                    onChange={(event) => updateStoneBrowseFilter("minCarat", event.target.value)}
                  />
                </Field>
                <Field label="Max carat">
                  <input
                    className="field-control"
                    type="number"
                    step="0.001"
                    value={stoneBrowseFilters.maxCarat}
                    onChange={(event) => updateStoneBrowseFilter("maxCarat", event.target.value)}
                  />
                </Field>
                <Field label="Min stone price">
                  <input
                    className="field-control"
                    type="number"
                    step="0.01"
                    value={stoneBrowseFilters.minPricePerCarat}
                    onChange={(event) => updateStoneBrowseFilter("minPricePerCarat", event.target.value)}
                  />
                </Field>
                <Field label="Max stone price">
                  <input
                    className="field-control"
                    type="number"
                    step="0.01"
                    value={stoneBrowseFilters.maxPricePerCarat}
                    onChange={(event) => updateStoneBrowseFilter("maxPricePerCarat", event.target.value)}
                  />
                </Field>
              </div>
              <div className="catalog-filter-actions">
                <InlineButton
                  disabled={stoneActiveFilterCount === 0}
                  onClick={() => {
                    setStoneBrowseFilters(blankStoneBrowseFilters);
                    setStonePage(1);
                    setStoneAssistNotice("");
                  }}
                >
                  Clear stone filters{stoneActiveFilterCount ? ` (${stoneActiveFilterCount})` : ""}
                </InlineButton>
              </div>
            </>
          }
          items={stones}
          total={stoneTotal}
          page={stonePage}
          totalPages={stoneTotalPages}
          pageSize={stonePageSize}
          loading={stoneLoading}
          error={stoneError}
          columns={["SKU", "Stone", "Shape", "Color", "Carat", "Size", "Cut / quality", "Stone price"]}
          selectedId={selectedStone?.stone_id ?? ""}
          suggestions={stoneSuggestions}
          onSearchChange={(value) => {
            setStoneSearch(value);
            setStonePage(1);
            setStoneAssistNotice("");
          }}
          onPageSizeChange={(value) => {
            setStonePageSize(value);
            setStonePage(1);
          }}
          onPrev={() => setStonePage((current) => current - 1)}
          onNext={() => setStonePage((current) => current + 1)}
          onSelect={(stone) => setSelectedStone(stone)}
          onSuggestionSelect={(stone) => {
            setStoneSearch(`${stone.name} ${stone.stone_id}`);
            setSelectedStone(stone);
          }}
          renderRow={(stone) => (
            <>
              <td>
                <strong>{stone.stone_id}</strong>
              </td>
              <td>{stone.name}</td>
              <td>{stone.shape}</td>
              <td>{stone.color || "Not set"}</td>
              <td>{stone.carat} ct</td>
              <td>{formatStoneSize(stone.min_size_mm, stone.max_size_mm) || "Not set"}</td>
              <td>{stone.quality || "Not set"}</td>
              <td>{money(stone.final_price)}</td>
            </>
          )}
          renderSuggestion={(stone) =>
            `${stone.stone_id} | ${stone.shape} | ${formatStoneSize(stone.min_size_mm, stone.max_size_mm) || "No size"} | ${money(stone.final_price)}`
          }
          detail={
            selectedStone ? (
              <>
                <div className="detail-heading">
                  <div>
                    <h3>{selectedStone.name}</h3>
                  </div>
                </div>
                <dl className="detail-grid detail-grid--catalog">
                  <DetailItem label="SKU" value={selectedStone.stone_id} />
                  <DetailItem label="Color" value={selectedStone.color} />
                  <DetailItem label="Quality" value={selectedStone.quality || "Not specified"} />
                  <DetailItem label="Size" value={formatStoneSize(selectedStone.min_size_mm, selectedStone.max_size_mm) || "Not specified"} />
                  <DetailItem label="Stone price" value={money(selectedStone.final_price)} />
                </dl>
                <div className="detail-action-row">
                  <Field label="Quantity">
                    <input
                      className="field-control"
                      type="number"
                      min="1"
                      step="1"
                      value={selectedStoneQuantity}
                      onChange={(event) => setSelectedStoneQuantity(event.target.value)}
                    />
                  </Field>
                  <button className="button button--block" type="button" onClick={() => attachStoneToProject(selectedStone)}>
                    Add to project
                  </button>
                </div>
              </>
            ) : (
              <p className="empty-state">Select a stone.</p>
            )
          }
        />
      </SectionCard>

      <SectionCard eyebrow="Settings" title="Setting catalog" compact>
        <CatalogPanel
          kind="setting"
          searchValue={settingSearch}
          searchPlaceholder="Search SKU, style, metal"
          searchActions={
            <InlineButton disabled={settingAssistLoading || !settingSearch.trim()} onClick={() => handleCatalogAssist("setting")}>
              {settingAssistLoading ? "Filtering..." : "AI filter"}
            </InlineButton>
          }
          searchNote={settingAssistNotice || undefined}
          filters={
            <>
              <div className="catalog-filter-grid">
                <Field label="Product ID">
                  <input
                    className="field-control"
                    value={settingBrowseFilters.productId}
                    onChange={(event) => updateSettingBrowseFilter("productId", event.target.value)}
                    placeholder="Shopify ID"
                  />
                </Field>
                <Field label="SKU">
                  <input
                    className="field-control"
                    value={settingBrowseFilters.settingId}
                    onChange={(event) => updateSettingBrowseFilter("settingId", event.target.value)}
                    placeholder="SRY..."
                  />
                </Field>
                <Field label="Style">
                  <input
                    className="field-control"
                    value={settingBrowseFilters.style}
                    onChange={(event) => updateSettingBrowseFilter("style", event.target.value)}
                    placeholder="Band, halo"
                  />
                </Field>
                <Field label="Metal">
                  <input
                    className="field-control"
                    value={settingBrowseFilters.metal}
                    onChange={(event) => updateSettingBrowseFilter("metal", event.target.value)}
                    placeholder="Gold, platinum"
                  />
                </Field>
                <Field label="Min complexity">
                  <input
                    className="field-control"
                    type="number"
                    step="1"
                    value={settingBrowseFilters.minComplexity}
                    onChange={(event) => updateSettingBrowseFilter("minComplexity", event.target.value)}
                  />
                </Field>
                <Field label="Max complexity">
                  <input
                    className="field-control"
                    type="number"
                    step="1"
                    value={settingBrowseFilters.maxComplexity}
                    onChange={(event) => updateSettingBrowseFilter("maxComplexity", event.target.value)}
                  />
                </Field>
                <Field label="Min weight (g)">
                  <input
                    className="field-control"
                    type="number"
                    step="0.1"
                    value={settingBrowseFilters.minWeightG}
                    onChange={(event) => updateSettingBrowseFilter("minWeightG", event.target.value)}
                  />
                </Field>
                <Field label="Max weight (g)">
                  <input
                    className="field-control"
                    type="number"
                    step="0.1"
                    value={settingBrowseFilters.maxWeightG}
                    onChange={(event) => updateSettingBrowseFilter("maxWeightG", event.target.value)}
                  />
                </Field>
                <Field label="Min labor">
                  <input
                    className="field-control"
                    type="number"
                    step="0.01"
                    value={settingBrowseFilters.minLaborCost}
                    onChange={(event) => updateSettingBrowseFilter("minLaborCost", event.target.value)}
                  />
                </Field>
                <Field label="Max labor">
                  <input
                    className="field-control"
                    type="number"
                    step="0.01"
                    value={settingBrowseFilters.maxLaborCost}
                    onChange={(event) => updateSettingBrowseFilter("maxLaborCost", event.target.value)}
                  />
                </Field>
                <Field label="Min base price">
                  <input
                    className="field-control"
                    type="number"
                    step="0.01"
                    value={settingBrowseFilters.minBasePrice}
                    onChange={(event) => updateSettingBrowseFilter("minBasePrice", event.target.value)}
                  />
                </Field>
                <Field label="Max base price">
                  <input
                    className="field-control"
                    type="number"
                    step="0.01"
                    value={settingBrowseFilters.maxBasePrice}
                    onChange={(event) => updateSettingBrowseFilter("maxBasePrice", event.target.value)}
                  />
                </Field>
              </div>
              <div className="catalog-filter-actions">
                <InlineButton
                  disabled={settingActiveFilterCount === 0}
                  onClick={() => {
                    setSettingBrowseFilters(blankSettingBrowseFilters);
                    setSettingPage(1);
                    setSettingAssistNotice("");
                  }}
                >
                  Clear setting filters{settingActiveFilterCount ? ` (${settingActiveFilterCount})` : ""}
                </InlineButton>
              </div>
            </>
          }
          items={settings}
          total={settingTotal}
          page={settingPage}
          totalPages={settingTotalPages}
          pageSize={settingPageSize}
          loading={settingLoading}
          error={settingError}
          columns={[
            "SKU",
            "Style",
            "Metal",
            "Complexity",
            "Gold 999",
            "Labor",
            "14K setting price",
          ]}
          selectedId={selectedSetting?.setting_id ?? ""}
          suggestions={settingSuggestions}
          onSearchChange={(value) => {
            setSettingSearch(value);
            setSettingPage(1);
            setSettingAssistNotice("");
          }}
          onPageSizeChange={(value) => {
            setSettingPageSize(value);
            setSettingPage(1);
          }}
          onPrev={() => setSettingPage((current) => current - 1)}
          onNext={() => setSettingPage((current) => current + 1)}
          onSelect={(setting) => setSelectedSetting(setting)}
          onSuggestionSelect={(setting) => {
            setSettingSearch(`${setting.style} ${setting.setting_id}`);
            setSelectedSetting(setting);
          }}
          renderRow={(setting) => (
            <>
              <td>
                <strong>{setting.setting_id}</strong>
              </td>
              <td>{setting.style}</td>
              <td>{setting.metal}</td>
              <td>{setting.complexity_level || "Unknown"}</td>
              <td>{setting.gold_weight_g} g</td>
              <td>{money(setting.labor_cost)}</td>
              <td>{money(setting.base_price)}</td>
            </>
          )}
          renderSuggestion={(setting) =>
            `${setting.setting_id} | ${setting.metal} | ${setting.gold_weight_g} g | ${money(setting.base_price)}`
          }
          detail={
            selectedSetting ? (
              <>
                <div className="detail-heading">
                  <div>
                    <h3>{selectedSetting.style}</h3>
                  </div>
                </div>
                <dl className="detail-grid detail-grid--catalog">
                  <DetailItem label="SKU" value={selectedSetting.setting_id} />
                  <DetailItem label="Metal" value={selectedSetting.metal} />
                  <DetailItem label="Complexity" value={selectedSetting.complexity_level ? String(selectedSetting.complexity_level) : "Unknown"} />
                  <DetailItem label="Gold 999 weight" value={`${selectedSetting.gold_weight_g} g`} />
                  <DetailItem label="Labor cost" value={money(selectedSetting.labor_cost)} />
                  <DetailItem label="14K setting price" value={money(selectedSetting.base_price)} />
                </dl>
                <button className="button" type="button" onClick={() => attachSettingToProject(selectedSetting)}>
                  Add to project
                </button>
              </>
            ) : (
              <p className="empty-state">Select a setting.</p>
            )
          }
        />
      </SectionCard>
    </>
  );
  const projectSection = (
    <SectionCard eyebrow="Listing" title="Custom listing">
      <div className="section-grid section-grid--two">
        <form className="stack" onSubmit={handleProjectSubmit}>
          <div className="detail-card detail-card--recall">
            <div className="detail-heading">
              <div>
                <h3>Recall by Shopify ID / URL</h3>
              </div>
              {recalledProduct ? <StatusPill>{recalledProduct.matched_by}</StatusPill> : null}
            </div>
            <div className="stack">
              <div className="recall-grid">
                <Field label="Shopify ID or product URL">
                  <input
                    className="field-control"
                    value={recallReference}
                    onChange={(event) => setRecallReference(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        void runRecall();
                      }
                    }}
                    placeholder="6563780034603 or https://www.capucinne.com/products/..."
                  />
                </Field>
                <div className="action-row action-row--end">
                  <button className="button" type="button" disabled={recallLoading} onClick={() => void runRecall()}>
                    {recallLoading ? "Recalling..." : "Recall setup"}
                  </button>
                </div>
              </div>
              {recallNotice ? <p className="inline-note">{recallNotice}</p> : null}
              {recalledProduct ? (
                <>
                  <dl className="detail-grid detail-grid--compact">
                    <DetailItem label="Shopify ID" value={recalledProduct.product_id} />
                    <DetailItem label="Handle" value={recalledProduct.product_handle || "Not set"} />
                    <DetailItem label="Title" value={recalledProduct.title || "Not set"} />
                    <DetailItem label="Variants" value={String(recalledProduct.variants.length)} />
                  </dl>
                  {recalledProduct.variants.length > 1 ? (
                    <Field label="Variant line">
                      <select
                        className="field-control"
                        value={recalledVariantKey}
                        onChange={(event) => {
                          setRecalledVariantKey(event.target.value);
                          applyRecalledVariant(recalledProduct, event.target.value);
                        }}
                      >
                        {recalledProduct.variants.map((variant) => (
                          <option key={variant.variant_key} value={variant.variant_key}>
                            {[variant.variant_sku, variant.metal, variant.setting_ids.join(", ")].filter(Boolean).join(" | ")}
                          </option>
                        ))}
                      </select>
                    </Field>
                  ) : null}
                  {activeRecalledVariant ? (
                    <div className="detail-block">
                      <h4>Recalled setup</h4>
                      <p className="detail-note">
                        {activeRecalledVariant.setting_ids.length
                          ? `Settings: ${activeRecalledVariant.setting_ids.join(", ")}`
                          : "No setting SKU found on this row."}
                      </p>
                      <div className="selection-token-list">
                        {activeRecalledVariant.stones.map((line) => (
                          <article key={`${line.role}-${line.stone_id}`} className="selection-token">
                            <div className="selection-token__meta">
                              <strong>{line.stone_id || "Missing stone SKU"}</strong>
                              <span>
                                {line.label} | Qty {line.quantity}
                                {line.measurements ? ` | ${line.measurements}` : ""}
                                {line.stone ? ` | ${line.stone.name}` : " | Missing from catalog"}
                              </span>
                            </div>
                          </article>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </>
              ) : null}
            </div>
          </div>
          <div className="project-selection-grid">
            <SelectionCard
              title="Stones in project"
              value={
                projectStoneRefs.length
                  ? `${projectStoneRefs.length} stone ${pluralize(projectStoneRefs.length, "type")} / ${totalProjectStoneQuantity} total`
                  : "No catalog stones attached"
              }
              note={projectStoneRefs.length ? summarizeSelection(projectStoneIds) : undefined}
            />
            <SelectionCard
              title="Settings in project"
              value={
                projectSettingRefs.length
                  ? `${projectSettingRefs.length} catalog ${pluralize(projectSettingRefs.length, "setting")} attached`
                  : "No catalog settings attached"
              }
              note={projectSettingRefs.length ? summarizeSelection(projectSettingIds) : undefined}
            />
          </div>
          <div className="selection-stack">
            <div className="selection-collection">
              <p className="field-label">Attached stones</p>
              {projectStoneRefs.length ? (
                <div className="selection-token-list">
                  {projectStoneRefs.map((line) => (
                    <article key={line.stone.stone_id} className="selection-token">
                      <div className="selection-token__meta">
                        <strong>{line.stone.name}</strong>
                        <span>{line.stone.stone_id} | {line.stone.shape} | {line.stone.carat} ct | Qty {line.quantity}</span>
                      </div>
                      <button className="inline-button selection-token__remove" type="button" onClick={() => removeStoneFromProject(line.stone.stone_id)}>
                        Remove
                      </button>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="inline-note">No stones attached.</p>
              )}
            </div>
            <div className="selection-collection">
              <p className="field-label">Attached settings</p>
              {projectSettingRefs.length ? (
                <div className="selection-token-list">
                  {projectSettingRefs.map((setting) => (
                    <article key={setting.setting_id} className="selection-token">
                      <div className="selection-token__meta">
                        <strong>{setting.style}</strong>
                        <span>{setting.setting_id} | {setting.metal} | {setting.gold_weight_g} g</span>
                      </div>
                      <button className="inline-button selection-token__remove" type="button" onClick={() => removeSettingFromProject(setting.setting_id)}>
                        Remove
                      </button>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="inline-note">No settings attached.</p>
              )}
            </div>
          </div>
          <div className="action-row action-row--compact">
            <InlineButton
              disabled={!projectStoneRefs.length}
              onClick={() => {
                setProjectStoneRefs([]);
                setProjectForm((current) => ({ ...current, stone_id: "" }));
              }}
            >
              Clear stones
            </InlineButton>
            <InlineButton
              disabled={!projectSettingRefs.length}
              onClick={() => {
                setProjectSettingRefs([]);
                setProjectForm((current) => ({ ...current, setting_id: "" }));
              }}
            >
              Clear settings
            </InlineButton>
          </div>
          <div className="form-grid">
            <Field label="Customer / project name"><input className="field-control" required value={projectForm.customer_name} onChange={(event) => setProjectForm({ ...projectForm, customer_name: event.target.value })} /></Field>
            <Field label="Target size (mm)"><input className="field-control" type="number" step="0.1" value={projectForm.target_size_mm} onChange={(event) => setProjectForm({ ...projectForm, target_size_mm: event.target.value })} /></Field>
            <Field label="Pure gold override (g)"><input className="field-control" type="number" step="0.1" value={projectForm.target_gold_weight_g} onChange={(event) => setProjectForm({ ...projectForm, target_gold_weight_g: event.target.value })} placeholder="Optional, used for metal reference only" /></Field>
            <Field label="Reference image URL"><input className="field-control" value={projectForm.reference_image_url} onChange={(event) => setProjectForm({ ...projectForm, reference_image_url: event.target.value })} /></Field>
            <Field label="Status"><select className="field-control" value={projectForm.status} onChange={(event) => setProjectForm({ ...projectForm, status: event.target.value })}><option value="open">open</option><option value="quoted">quoted</option><option value="closed">closed</option></select></Field>
            <Field label="Created by"><input className="field-control" value={projectForm.created_by} onChange={(event) => setProjectForm({ ...projectForm, created_by: event.target.value })} /></Field>
          </div>
          <div className="action-row"><button className="button" disabled={isPending} type="submit">Save custom listing</button>{projectNotice ? <p className="inline-note">{projectNotice}</p> : null}</div>
        </form>

        <div className="detail-stack">
          <div className="detail-card">
            <div className="detail-heading"><div><h3>Costs and quote</h3></div></div>
            <div className="quote-grid">
              <QuoteItem label="Stone total" value={money(projectQuote.stoneCost)} />
              <QuoteItem label="14K setting price" value={money(projectQuote.basePrice)} />
              <QuoteItem label="B2C total" value={money(projectQuote.catalogSubtotal)} />
              <QuoteItem label="Estimated quote 14K" value={money(projectQuote.estimatedQuote14k)} highlight />
              <QuoteItem label="Estimated quote 18K" value={money(projectQuote.estimatedQuote18k)} highlight />
            </div>
            <dl className="detail-grid detail-grid--compact">
              <DetailItem label="Gold 999 rate" value={money(projectQuote.gold_price_per_gram)} />
              <DetailItem label="14K pure gold weight" value={`${projectQuote.gold_weight_g} g`} />
              <DetailItem label="Stones" value={`${projectQuote.stone_count} total / ${projectStoneRefs.length} SKU`} />
              <DetailItem label="Settings" value={String(projectQuote.setting_count)} />
            </dl>
          </div>
          <div className="table-card">
            <div className="table-card__header"><h3>Listings</h3><span>{projects.length}</span></div>
            <table className="data-table">
              <thead><tr><th>Listing</th><th>Status</th><th>B2C total</th><th>Quotes</th></tr></thead>
              <tbody>
                {projects.slice(0, 6).map((project) => (
                  <tr key={project.inquiry_id}>
                    <td><strong>{project.customer_name}</strong><span>{project.inquiry_id}</span></td>
                    <td>{project.status}</td>
                    <td>{money(project.estimated_material_cost)}</td>
                    <td>
                      <strong>14K {money(project.estimated_quote)}</strong>
                      <span>18K {money(project.estimated_quote_18k ?? project.estimated_quote)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </SectionCard>
  );

  const aiSection = (
    <SectionCard eyebrow="Approximation" title="AI approximation">
      <div className="section-grid section-grid--two">
        <form className="stack" onSubmit={handleValuationSubmit}>
          <Field label="Description">
            <textarea
              className="field-control field-control--textarea field-control--textarea-lg"
              rows={8}
              required
              value={valuationForm.description}
              onChange={(event) => setValuationForm({ ...valuationForm, description: event.target.value })}
              placeholder="Describe the piece in full detail, as if you were briefing a jewelry expert directly. Include metal, weight, style, proportions, stone arrangement, dimensions, finish, shape, inspiration, era references, construction clues, and anything else that helps Gemini understand the piece contextually."
            />
          </Field>
          <div className="form-grid">
            <Field label="Reference image URL">
              <input
                className="field-control"
                value={valuationForm.reference_image_url}
                onChange={(event) => setValuationForm({ ...valuationForm, reference_image_url: event.target.value })}
                placeholder="https://..."
              />
            </Field>
            <Field label="Attach image">
              <input
                className="field-control field-control--file"
                type="file"
                accept="image/*"
                onChange={(event) => {
                  void handleValuationImageChange(event);
                }}
              />
            </Field>
          </div>
          {valuationForm.reference_image_url || valuationForm.image_data_url ? (
            <div className="valuation-media-grid">
              <MediaPreview src={valuationForm.reference_image_url} alt="Reference image preview" />
              <MediaPreview src={valuationForm.image_data_url} alt="Uploaded image preview" />
            </div>
          ) : null}
          <div className="action-row">
            <button className="button" disabled={valuationLoading || isPending} type="submit">
              {valuationLoading ? "Thinking..." : "Run approximation"}
            </button>
            {valuationNotice ? <p className="inline-note">{valuationNotice}</p> : null}
          </div>
        </form>

        <div className="detail-stack">
          <div className="detail-card">
            <div className="detail-heading">
              <div>
                <h3>Latest approximation</h3>
              </div>
              {valuationLoading || valuationFollowUpLoading ? (
                <StatusPill tone="rose">thinking</StatusPill>
              ) : valuationResult ? (
                <StatusPill>{valuationResult.provider}</StatusPill>
              ) : null}
            </div>
            <div key={valuationLoading ? "thinking" : valuationResult?.valuation_id ?? "empty"} className="ai-response-frame">
            {valuationLoading ? (
              <>
                <div className="detail-block">
                  <h4>Request</h4>
                  <p className="detail-note">{valuationForm.description}</p>
                  {valuationForm.reference_image_url ? (
                    <p className="detail-note">
                      <a className="link-inline" href={valuationForm.reference_image_url} target="_blank" rel="noreferrer">
                        {valuationForm.reference_image_url}
                      </a>
                    </p>
                  ) : null}
                  {(valuationForm.reference_image_url || valuationForm.image_data_url) ? (
                    <div className="valuation-media-grid">
                      <MediaPreview src={valuationForm.reference_image_url} alt="Reference image preview" />
                      <MediaPreview src={valuationForm.image_data_url} alt="Uploaded image preview" />
                    </div>
                  ) : null}
                </div>
                <div className="thinking-panel">
                  <div className="thinking-panel__status">
                    <span>Loading response from Gemini</span>
                    <div className="thinking-dots" aria-hidden="true">
                      <span className="thinking-dot" />
                      <span className="thinking-dot" />
                      <span className="thinking-dot" />
                    </div>
                  </div>
                  <p className="detail-note">Reading the brief, image context, and catalog references to assemble an estimate.</p>
                  <div className="thinking-lines" aria-hidden="true">
                    <span className="thinking-line thinking-line--long" />
                    <span className="thinking-line thinking-line--mid" />
                    <span className="thinking-line thinking-line--short" />
                  </div>
                </div>
              </>
            ) : valuationResult ? (
              <>
                <div className="detail-block">
                  <h4>Request</h4>
                  <p className="detail-note">{valuationResult.description}</p>
                  {valuationResult.reference_image_url ? (
                    <p className="detail-note">
                      <a className="link-inline" href={valuationResult.reference_image_url} target="_blank" rel="noreferrer">
                        {valuationResult.reference_image_url}
                      </a>
                    </p>
                  ) : null}
                  <div className="valuation-media-grid">
                    <MediaPreview src={valuationResult.reference_image_url} alt="Reference image" />
                    <MediaPreview src={valuationResult.image_data_url} alt="Uploaded image" />
                  </div>
                </div>
                <div className="quote-grid">
                  <QuoteItem label="Stone subtotal" value={money(valuationResult.estimated_stone_total)} />
                  <QuoteItem label="Setting subtotal" value={money(valuationResult.estimated_setting_total)} />
                  <QuoteItem label="Formula estimate" value={money(valuationResult.estimated_formula_total || midpoint(valuationResult.estimated_value_low, valuationResult.estimated_value_high))} highlight />
                </div>
                <div className="detail-block">
                  <h4>Gemini response</h4>
                  <p className="detail-note detail-note--strong">{valuationResult.pricing_summary}</p>
                  <p className="detail-note">{valuationResult.reasoning}</p>
                  <p className="detail-note">{valuationResult.recommended_next_step}</p>
                </div>
                {valuationFollowUpMessages.length ? (
                  <div className="detail-block">
                    <h4>Refinements</h4>
                    <div className="conversation-thread">
                      {valuationFollowUpMessages.map((message) => (
                        <article
                          key={message.message_id}
                          className={`conversation-message conversation-message--${message.role}`}
                        >
                          <span className="conversation-message__role">{message.role === "user" ? "You" : "Gemini"}</span>
                          <p>{message.role === "assistant" ? buildAssistantMessagePreview(message) : message.content}</p>
                        </article>
                      ))}
                    </div>
                  </div>
                ) : null}
                <form className="stack" onSubmit={handleValuationFollowUpSubmit}>
                  <Field label="Continue this approximation">
                    <textarea
                      className="field-control field-control--textarea"
                      rows={3}
                      value={valuationFollowUp}
                      onChange={(event) => setValuationFollowUp(event.target.value)}
                      placeholder="Same ring, but all lab dia. Keep the setting and replace the side stones."
                    />
                  </Field>
                  <div className="action-row action-row--compact">
                    <button
                      className="button"
                      type="submit"
                      disabled={valuationFollowUpLoading || !valuationResult}
                    >
                      {valuationFollowUpLoading ? "Refining..." : "Continue with Gemini"}
                    </button>
                    <InlineButton onClick={() => loadValuationIntoForm(valuationResult)}>Load into approximation</InlineButton>
                    <InlineButton onClick={() => openValuationDetails(valuationResult)}>Open details</InlineButton>
                  </div>
                </form>
              </>
            ) : (
              <p className="empty-state">No approximation yet.</p>
            )}
            </div>
          </div>
          <div className="table-card">
            <div className="table-card__header"><h3>Log</h3><span>{valuations.length}</span></div>
            <table className="data-table">
              <thead><tr><th>Description</th><th>Estimate</th><th>Provider</th><th>Updated</th></tr></thead>
              <tbody>
                {valuations.slice(0, 6).map((valuation) => (
                  <tr key={valuation.valuation_id} onClick={() => openValuationDetails(valuation)}>
                    <td>
                      <strong>{excerpt(valuation.description)}</strong>
                      <span>{valuation.valuation_id}</span>
                    </td>
                    <td>{money(valuation.estimated_formula_total || midpoint(valuation.estimated_value_low, valuation.estimated_value_high))}</td>
                    <td>{valuation.provider}</td>
                    <td>{stamp(valuation.updated_at || valuation.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </SectionCard>
  );

  return (
    <main className="dashboard-shell">
      {toasts.length ? (
        <div className="toast-stack" aria-live="polite" aria-atomic="true">
          {toasts.map((toast) => (
            <article key={toast.id} className={`toast toast--${toast.tone}`}>
              <strong>{toast.tone === "success" ? "Added" : "Not added"}</strong>
              <span>{toast.message}</span>
            </article>
          ))}
        </div>
      ) : null}
      {catalogSections}
      {projectSection}
      {aiSection}
      {valuationModal ? (
        <div className="modal-backdrop" role="presentation" onClick={() => setValuationModal(null)}>
          <section className="modal-card" role="dialog" aria-modal="true" aria-label="Approximation details" onClick={(event) => event.stopPropagation()}>
            <div className="detail-heading">
              <div>
                <p className="eyebrow">Approximation details</p>
                <h3>{excerpt(valuationModal.description, 120)}</h3>
              </div>
              <StatusPill>{valuationModal.provider}</StatusPill>
            </div>
            <div className="detail-block">
              <h4>Request</h4>
              <p className="detail-note">{valuationModal.description}</p>
              {valuationModal.reference_image_url ? (
                <p className="detail-note">
                  <a className="link-inline" href={valuationModal.reference_image_url} target="_blank" rel="noreferrer">
                    {valuationModal.reference_image_url}
                  </a>
                </p>
              ) : null}
              <div className="valuation-media-grid">
                <MediaPreview src={valuationModal.reference_image_url} alt="Reference image" />
                <MediaPreview src={valuationModal.image_data_url} alt="Uploaded image" />
              </div>
              <p className="detail-note">Updated {stamp(valuationModal.updated_at || valuationModal.created_at)}</p>
            </div>
            <div className="quote-grid">
              <QuoteItem label="Stone subtotal" value={money(valuationModal.estimated_stone_total)} />
              <QuoteItem label="Setting subtotal" value={money(valuationModal.estimated_setting_total)} />
              <QuoteItem label="Formula estimate" value={money(valuationModal.estimated_formula_total || midpoint(valuationModal.estimated_value_low, valuationModal.estimated_value_high))} highlight />
            </div>
            <div className="detail-block">
              <h4>Gemini response</h4>
              <p className="detail-note detail-note--strong">{valuationModal.pricing_summary}</p>
              <p className="detail-note">{valuationModal.reasoning}</p>
              <p className="detail-note">{valuationModal.recommended_next_step}</p>
            </div>
            {valuationModal.messages.length > 2 ? (
              <div className="detail-block">
                <h4>Refinements</h4>
                <div className="conversation-thread">
                  {valuationModal.messages.slice(2).map((message) => (
                    <article
                      key={message.message_id}
                      className={`conversation-message conversation-message--${message.role}`}
                    >
                      <span className="conversation-message__role">{message.role === "user" ? "You" : "Gemini"}</span>
                      <p>{message.role === "assistant" ? buildAssistantMessagePreview(message) : message.content}</p>
                    </article>
                  ))}
                </div>
              </div>
            ) : null}
            <div className="modal-actions">
              <button className="button" type="button" onClick={() => loadValuationIntoForm(valuationModal)}>
                Load into approximation
              </button>
              <InlineButton onClick={() => setValuationModal(null)}>Close</InlineButton>
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}
