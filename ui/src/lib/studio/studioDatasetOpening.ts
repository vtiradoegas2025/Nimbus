import { datasetSourceIdentity, type LocalDatasetSource } from "@/lib/dataset/datasetSources";
import { MAX_ENSEMBLE_MEMBERS, type EnsembleMemberInput } from "@/lib/comparison/ensembleAlignment";

export function admitEnsembleMember(
  primary: LocalDatasetSource | null,
  members: readonly EnsembleMemberInput[],
  candidate: EnsembleMemberInput,
): EnsembleMemberInput[] {
  if (candidate.steps.length === 0) throw new Error("Ensemble member has no complete plottable frames.");
  if (members.length + 1 >= MAX_ENSEMBLE_MEMBERS) throw new Error(`Interactive ensembles are bounded to ${MAX_ENSEMBLE_MEMBERS} members.`);
  const identity = datasetSourceIdentity(candidate.source);
  if (primary && identity === datasetSourceIdentity(primary)
      || members.some(member => datasetSourceIdentity(member.source) === identity)) {
    throw new Error("That dataset is already in the ensemble group.");
  }
  return [...members, candidate];
}

export async function openRequiredDatasetSources(input: {
  sources: readonly { role: "primary" | "comparison" | "ensemble"; source: LocalDatasetSource }[];
  current: () => boolean;
  load: (role: "primary" | "comparison" | "ensemble", source: LocalDatasetSource) => Promise<boolean | undefined>;
}): Promise<boolean> {
  for (const { role, source } of input.sources) {
    if (!input.current()) return false;
    let opened: boolean | undefined;
    try { opened = await input.load(role, source); }
    catch (reason) {
      if (!input.current()) return false;
      throw reason;
    }
    if (!input.current()) return false;
    if (!opened) throw new Error(`The ${role} source '${source.locator.path}' could not be reopened. Reconnect or relink the data, then try again.`);
  }
  return input.current();
}
