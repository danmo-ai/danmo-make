/** Structural ControlNets attach from advanced params. They are not generation models.
 *  standalone_edit cards (Fill) stay selectable when their actions match the mode. */
export function isStructuralComposerControlNet(
  row: Record<string, unknown> | null | undefined,
): boolean {
  if (!row || row.category !== 'controlnets') return false;
  return String(row.binding || '').trim() !== 'standalone_edit';
}
