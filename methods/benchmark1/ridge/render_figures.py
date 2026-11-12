#!/usr/bin/env python3
"""Render exact broad-label Benchmark 1 panels from local verified masks.

The former KODAMA–Bora spatial group mask is unavailable in the curated local
store, so this renderer deliberately leaves its panel blank instead of
approximating labels from a lossy overlay.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SITE = Path('/tmp/bora-analysis-bw106-v2/site')
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
SUPERVISED = HERE / 'DEVELOPMENT_CONTOUR_COMPLETE.json'
BASE = DATA / 'benchmark1-supervised-mask-overlay-score-2026-11-10-v1/SCORES_AND_FIGURES.json'
GEOM = DATA / 'benchmark1-supervised-virchow2-response-2026-11-10-v1'
HARM = ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json'
VISTA = ROOT / 'benchmark/comparators/VISTA_BRCA_CLASS_MAP_PREDECLARED_2026-11-07.json'
VISTA_NAMES = ('tumor', 'stroma', 'lymphocytic_infiltrate', 'necrosis_or_debris',
               'blood', 'metaplasia_NOS', 'fat', 'plasma_cells',
               'other_immune_infiltrate', 'lymphatics', 'blood_vessel',
               'normal_acinus_or_duct', 'dcis')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError('SHA mismatch: ' + str(path))
    return path


def nn(mask, size):
    if mask.shape == size[::-1]:
        return mask
    return np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))


def overlay(he, mask, support, palette):
    pixels = np.asarray(he, np.uint8)
    result = np.rint(.45 * pixels + .55 * palette[mask]).astype(np.uint8)
    result[(mask == 0) | ~support] = pixels[(mask == 0) | ~support]
    return Image.fromarray(result, 'RGB')


def main(out, only_id):
    if out.exists():
        raise FileExistsError(out)
    site_cases = json.loads((SITE / 'benchmark1-comparators/cases.json').read_text())['cases']
    base_cases = {r['case']: r for r in json.loads(BASE.read_text())['cases']}
    frozen = json.loads(FROZEN.read_text())
    inputs = {r['case']: r for r in frozen['rows'] if r['split'] == 'validation'}
    predicted = {r['case']: r for r in json.loads(SUPERVISED.read_text())['cases']}
    assert len(site_cases) == len(base_cases) == len(inputs) == len(predicted) == 87
    ontology = json.loads(ONTOLOGY.read_text())
    lut = np.zeros(256, np.uint8)
    for old, new in ontology['old_to_new'].items():
        if new is not None:
            lut[int(old)] = int(new)
    role_to_new = {role: ontology['old_to_new'][old]
                   for old, role in frozen['classes'].items()}
    harmony = json.loads(HARM.read_text())['mapping']
    vista_map = {r['vista_name']: r for r in json.loads(VISTA.read_text())['matched']}
    colors = json.loads((SITE / 'data.json').read_text())
    palette = np.asarray([colors['colours'][0]] + [
        colors['colours'][colors['roles'].index(frozen['classes'][str(i)]) + 1]
        for i in (1, 3, 5, 7, 8, 9, 11, 12)], np.uint8)
    rows = [r for r in site_cases if only_id is None or r['id'] == only_id]
    if not rows:
        raise ValueError('Unknown case')
    figures = out / 'figures'
    figures.mkdir(parents=True)
    font = ImageFont.truetype(FONT, 17)
    head = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 15)
    records = []
    for site in rows:
        case, cid = site['case'], site['id']
        source, baseline, sup = inputs[case], base_cases[case], predicted[case]
        assert sup['id'] == baseline['id'] == cid
        spath = checked(Path(sup['output_path']), sup['output_sha256'])
        with Image.open(spath) as im:
            supervised = np.asarray(im, np.uint8)
        wh = supervised.shape[::-1]
        geom = json.loads((GEOM / cid / 'RESPONSE_COMPLETE.json').read_text())
        assert wh == tuple(geom['combined_mask']['display_size_wh'])
        he_path = checked(Path(source['calibrated_image_path']), source['calibrated_image_sha256'])
        with Image.open(he_path) as im:
            he = im.convert('RGB').crop((0, 0, *geom['working_size_wh'])).resize(wh, Image.Resampling.LANCZOS)
        sup_path = checked(SITE / 'benchmark1-comparators' / site['he_background_support']['image'],
                           site['he_background_support']['sha256'])
        with Image.open(sup_path) as im:
            tissue = nn(np.asarray(im.getchannel('A'), np.uint8), wh) > 0
        expert_path = checked(Path(baseline['source_mask_path']), baseline['source_mask_sha256'])
        with Image.open(expert_path) as im:
            raw = np.asarray(im.resize(wh, Image.Resampling.NEAREST))
        expert = np.zeros(raw.shape, np.uint8)
        for code in np.unique(raw):
            spec = harmony[site['dataset']].get(str(int(code)))
            if spec and spec['use'] == 'score':
                new = role_to_new.get(spec.get('canonical_role'))
                if new is not None:
                    expert[raw == code] = int(new)
        masks = {'Expert reference': expert, 'Supervised Bora': supervised}
        for name, key, subpath in (
            ('Class-aware U-Net', 'unet', 'comparators-2026-11-07-v3/results/unet-balanced-v3/predictions'),
            ('DeepLabV3+', 'deeplabv3plus', 'comparators-deeplabv3plus-2026-11-08-v1/results/deeplabv3plus-v1/predictions')):
            path = checked(DATA / subpath / (case + '.npz'), site['source_sha256'][key])
            with np.load(path, allow_pickle=False) as z:
                masks[name] = lut[nn(z['prediction'].astype(np.uint8), wh)]
        vista_path = checked(DATA / 'comparators-2026-11-07-v1/results/vista-v2/cropped-to-true-frame' / (case + '.png'),
                             site['source_sha256']['vista'])
        with Image.open(vista_path) as im:
            vista = nn(np.asarray(im, np.uint8), wh)
        mapped = np.zeros(vista.shape, np.uint8)
        for code in np.unique(vista):
            if not (1 <= code <= len(VISTA_NAMES)):
                continue
            role = vista_map[VISTA_NAMES[code - 1]]
            if 'applies_to' in role and site['dataset'] not in role['applies_to']:
                continue
            new = role_to_new.get(role['bora_role'])
            if new is not None:
                mapped[vista == code] = int(new)
        masks['VISTA-PATH'] = mapped
        med_path = checked(DATA / 'comparators-2026-11-07-v1/results/medsam-guided-v1' / (case + '.npz'),
                           site['source_sha256']['medsam'])
        with np.load(med_path, allow_pickle=False) as z:
            components = {name: nn(z[name].astype(np.uint8), wh) > 0
                          for name in ('tumor', 'stroma', 'necrosis')}
        overlap = sum(x.astype(np.uint8) for x in components.values())
        med = np.zeros(raw.shape, np.uint8)
        for name, new in (('tumor', 1), ('stroma', 2), ('necrosis', 3)):
            med[components[name] & (overlap == 1)] = new
        masks['MedSAM'] = med
        canvas = Image.new('RGB', (1904, 1120), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 17), f"{site['dataset'].upper()} · {site['source_group']} · {cid}", font=head, fill='#17343d')
        draw.text((28, 52), 'Eight broad classes · development validation', font=font, fill='#42565c')
        panels = [('Original H&E', he), ('Expert reference', overlay(he, expert, np.ones(expert.shape, bool), palette))]
        panels += [(name, overlay(he, mask, tissue, palette)) for name, mask in masks.items() if name != 'Expert reference']
        assert len(panels) == 7
        shown = {int(value) for value in np.unique(expert) if value > 0}
        for name, mask in masks.items():
            if name == 'Expert reference':
                continue
            ids, counts = np.unique(mask[tissue], return_counts=True)
            shown |= {int(i) for i, count in zip(ids, counts) if i > 0 and count >= .01 * tissue.sum()}
        shown = sorted(shown)
        assert shown and set(shown) <= set(range(1, 9))
        for i, (name, image) in enumerate(panels):
            x, y = 28 + (i % 4) * 466, 105 + (i // 4) * 398
            draw.text((x, y), name, font=font, fill='#17343d')
            tile = ImageOps.contain(image, (450, 330), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x + (450 - tile.width) // 2, y + 34 + (330 - tile.height) // 2))
            draw.rectangle((x, y + 33, x + 450, y + 364), outline='#cbd8dd', width=1)
        draw.text((1426, 537), 'KODAMA–Bora + fastPLS', font=font, fill='#17343d')
        draw.text((1426, 577), 'Broad-label score from the', font=small, fill='#42565c')
        draw.text((1426, 602), 'frozen full confusion matrix.', font=small, fill='#42565c')
        draw.text((1426, 627), 'Spatial group mask unavailable', font=small, fill='#42565c')
        draw.text((1426, 652), 'in the curated local cache.', font=small, fill='#42565c')
        draw.text((1426, 677), 'No replacement overlay inferred', font=small, fill='#42565c')
        draw.text((1426, 702), 'from the former WebP image.', font=small, fill='#42565c')
        draw.line((28, 902, 1876, 902), fill='#cbd8dd', width=2)
        draw.text((28, 920), 'Classes present in expert or ≥1% of a displayed prediction', font=font, fill='#17343d')
        for i, value in enumerate(shown):
            x, y = 28 + (i % 4) * 466, 964 + (i // 4) * 38
            draw.rectangle((x, y, x + 18, y + 18), fill=tuple(map(int, palette[value])), outline='#617078')
            draw.text((x + 27, y - 1), ontology['classes'][str(value)]['name'].replace('_', ' ').capitalize(),
                      font=small, fill='#17343d')
        draw.text((28, 1060), 'Five rare tissue categories are unscored. All model overlays use the same Bora tissue support.',
                  font=small, fill='#42565c')
        target = figures / (cid + '.webp')
        canvas.save(target, 'WEBP', quality=85, method=6)
        records.append({'id': cid, 'case': case, 'image_sha256': sha(target),
                        'image_bytes': target.stat().st_size, 'expert_sha256': baseline['source_mask_sha256'],
                        'supervised_sha256': sup['output_sha256'],
                        'support_sha256': site['he_background_support']['sha256'],
                        'legend_class_ids': shown, 'kodama_spatial_mask_available': False})
        print('RENDERED', len(records), '/', len(rows), cid, flush=True)
    receipt = {'status': 'BROAD_LABEL_VALIDATION_FIGURES_RENDERED_WITH_KODAMA_SPATIAL_GAP',
               'case_count': len(records), 'ontology_sha256': sha(ONTOLOGY),
               'source_code_sha256': sha(Path(__file__)), 'cases': records,
               'pm016_included': False, 'beetle_included': False}
    (out / 'RENDERED.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--only-id')
    args = parser.parse_args()
    main(args.out, args.only_id)
