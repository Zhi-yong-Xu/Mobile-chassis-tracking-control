#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_mjcf.py —— 把带 <include> 的 MuJoCo MJCF 模型合并成一个独立的 XML 文件

用法:
    python merge_mjcf.py summit_xls.xml                    # -> summit_xls_merged.xml
    python merge_mjcf.py summit_xls.xml -o single.xml      # 指定输出名
    python merge_mjcf.py summit_xls.xml --copy-assets ./assets_out
                                                           # 同时收集 mesh/texture 资源

注意:
    - STL/OBJ 等 mesh 是二进制文件, 无法"塞进"XML 文本, 只能靠路径引用;
      如需目录级自包含, 请加 --copy-assets.
    - 若模型在 <compiler> 中使用了 meshdir/texturedir, 建议先删掉再合并.
"""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

# file 属性表示"外部资源路径"的 MJCF 标签
ASSET_TAGS = {'mesh', 'skin', 'hfield', 'texture'}


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def make_parser():
    """尽量保留 XML 注释 / 处理指令 (Python >= 3.8)"""
    try:
        return ET.XMLParser(
            target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    except TypeError:                       # 老版本 Python 退化为默认解析器
        return ET.XMLParser()


def parse_file(path):
    return ET.parse(path, parser=make_parser())


def to_posix(p):
    return p.replace(os.sep, '/')


# --------------------------------------------------------------------------- #
# 核心逻辑: 递归展开 <include>
# --------------------------------------------------------------------------- #
def expand(elem, cur_dir, main_dir, collect=None):
    """
    递归展开 elem 子树中的 <include>, 并把资源相对路径改写为相对 main_dir.

    cur_dir : 当前文件所在目录 (被包含文件里的相对路径原来相对它解析)
    main_dir: 最终输出 XML 所在目录 (路径改写目标)
    collect : 若不为 None, 收集 (资源绝对路径, 相对 main_dir 的新路径)
    """
    for child in list(elem):

        # ---------- 1) <include file="..."/> ----------
        if child.tag == 'include':
            f = child.get('file')
            if not f:
                raise ValueError('<include> 缺少 file 属性')

            inc_path = os.path.normpath(os.path.join(cur_dir, f))
            if not os.path.isfile(inc_path):
                raise FileNotFoundError(f'找不到被包含文件: {inc_path}')

            inc_dir = os.path.dirname(inc_path)
            sub_root = parse_file(inc_path).getroot()

            # 先递归处理子文件内部的 include / 资源路径
            expand(sub_root, inc_dir, main_dir, collect)

            # 用子文件根元素的"子节点"原地替换 <include> 标签
            idx = list(elem).index(child)
            elem.remove(child)
            for offset, node in enumerate(sub_root):
                elem.insert(idx + offset, node)
            continue    # 这些节点已经处理过

        # ---------- 2) 改写资源相对路径 ----------
        if child.tag in ASSET_TAGS and child.get('file'):
            raw = child.get('file')
            if not os.path.isabs(raw):
                # 先按"当前被包含文件"目录找, 找不到再按主文件目录找
                cand1 = os.path.normpath(os.path.join(cur_dir, raw))
                cand2 = os.path.normpath(os.path.join(main_dir, raw))
                real = cand1 if os.path.isfile(cand1) else \
                       (cand2 if os.path.isfile(cand2) else None)

                if real is None:
                    print(f'[警告] 找不到资源: {raw} '
                          f'(标签 <{child.tag}>, 目录 {cur_dir})', file=sys.stderr)
                else:
                    rel = to_posix(os.path.relpath(real, main_dir))
                    child.set('file', rel)
                    if collect is not None:
                        collect.append((real, rel))

        # ---------- 3) 向下递归 ----------
        expand(child, cur_dir, main_dir, collect)


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def serialize(tree, out_path):
    root = tree.getroot()
    try:                                    # Python >= 3.9
        ET.indent(tree, space='  ')
        data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    except AttributeError:                  # Python 3.8 退化用 minidom
        from xml.dom import minidom
        data = minidom.parseString(
            ET.tostring(root)).toprettyxml(indent='  ', encoding='utf-8')
    with open(out_path, 'wb') as fp:
        fp.write(data)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='把 MJCF <include> 合并成单个 XML')
    ap.add_argument('input', help='主模型文件, 例如 summit_xls.xml')
    ap.add_argument('-o', '--output', help='输出文件 (默认 <input>_merged.xml)')
    ap.add_argument('--copy-assets', metavar='DIR',
                    help='把引用到的 mesh/texture 等资源收集到该目录')
    args = ap.parse_args()

    in_path = os.path.abspath(args.input)
    main_dir = os.path.dirname(in_path)

    if not os.path.isfile(in_path):
        sys.exit(f'输入文件不存在: {in_path}')

    out_path = os.path.abspath(args.output) if args.output \
               else os.path.splitext(in_path)[0] + '_merged.xml'

    collect = [] if args.copy_assets else None
    if args.copy_assets:
        os.makedirs(os.path.abspath(args.copy_assets), exist_ok=True)

    tree = parse_file(in_path)
    root = tree.getroot()
    if root.tag != 'mujoco':
        print(f'[提示] 根元素是 <{root.tag}> 而不是 <mujoco>', file=sys.stderr)

    expand(root, main_dir, main_dir, collect)
    serialize(tree, out_path)
    print(f'[完成] 合并后的模型: {out_path}')

    if collect is not None:
        for src, rel in collect:
            dst = os.path.join(os.path.abspath(args.copy_assets), rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f'       资源 {src}  ->  {dst}')
        print(f'[完成] 共收集 {len(collect)} 个资源文件')


if __name__ == '__main__':
    main()
