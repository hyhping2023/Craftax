"""子 agent 提议的分级/复合任务（经校验后合并生成，勿手改）。

来源：task_graph_agents/*_proposals.json，经
craftax.tasks.agent_spec.validate_proposals 校验通过后由
scripts/build_task_graph.py --merge 生成。如需修改，调整 proposals 后重新生成。
"""
from __future__ import annotations

from typing import Any, Dict, List

from craftax.contracts import TaskSpec
from craftax.tasks.base import BaseTaskAdapter

TASK_VERSION = '1.0.0'

# (task_id, instruction, objective, success_predicate, dependencies)
_HIERARCHY_DEFS: List[Dict[str, Any]] = [
    {
        "task_id": 'native.basic_combat_training',
        "instruction": 'Arm yourself and defeat your first enemy. / 武装自己并击败第一个敌人。',
        "objective": '制作一把木剑并击败任意敌人（任意 DEFEAT_* 击杀成就达成）。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'MAKE_WOOD_SWORD'},
                        {'type': 'or',
                         'predicates': [{'type': 'achievement', 'name': 'DEFEAT_ZOMBIE'},
                                        {'type': 'achievement', 'name': 'DEFEAT_SKELETON'},
                                        {'type': 'achievement', 'name': 'DEFEAT_GNOME_WARRIOR'},
                                        {'type': 'achievement', 'name': 'DEFEAT_GNOME_ARCHER'},
                                        {'type': 'achievement', 'name': 'DEFEAT_ORC_SOLIDER'},
                                        {'type': 'achievement', 'name': 'DEFEAT_ORC_MAGE'},
                                        {'type': 'achievement', 'name': 'DEFEAT_LIZARD'},
                                        {'type': 'achievement', 'name': 'DEFEAT_KOBOLD'},
                                        {'type': 'achievement', 'name': 'DEFEAT_TROLL'},
                                        {'type': 'achievement', 'name': 'DEFEAT_DEEP_THING'},
                                        {'type': 'achievement', 'name': 'DEFEAT_PIGMAN'},
                                        {'type': 'achievement',
                                         'name': 'DEFEAT_FIRE_ELEMENTAL'},
                                        {'type': 'achievement', 'name': 'DEFEAT_FROST_TROLL'},
                                        {'type': 'achievement', 'name': 'DEFEAT_ICE_ELEMENTAL'},
                                        {'type': 'achievement',
                                         'name': 'DEFEAT_NECROMANCER'}]}]},
        "dependencies": ['native.craft_wood_sword', 'native.defeat_enemy'],
    },
    {
        "task_id": 'native.build_home_base',
        "instruction": 'Build a home base: place a table, a furnace and a torch. / 搭建家园基地：放置桌子、熔炉与火把。',
        "objective": '在出生地附近放置桌子、熔炉与火把（三者 PLACE_* 成就全部达成），完成基础营地建设。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'PLACE_TABLE'},
                        {'type': 'achievement', 'name': 'PLACE_FURNACE'},
                        {'type': 'achievement', 'name': 'PLACE_TORCH'}]},
        "dependencies": ['native.place_table', 'native.place_furnace', 'native.place_torch'],
    },
    {
        "task_id": 'native.clear_surface_threats',
        "instruction": 'Clear the surface threats. Defeat both a zombie and a skeleton. / 清除地表威胁。分别击败一只僵尸和一只骷髅。',
        "objective": '同时达成 DEFEAT_ZOMBIE 与 DEFEAT_SKELETON，清除地表最常见的两种夜间敌人，作为战斗类别的入门目标。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'DEFEAT_ZOMBIE'},
                        {'type': 'achievement', 'name': 'DEFEAT_SKELETON'}]},
        "dependencies": ['native.defeat_zombie', 'native.defeat_skeleton'],
    },
    {
        "task_id": 'native.collect_all_ores',
        "instruction": 'Collect every ore and gem. / 收集全部矿石与宝石。',
        "objective": '同时收集煤、铁、钻石、蓝宝石与红宝石（COLLECT_COAL / COLLECT_IRON / COLLECT_DIAMOND / COLLECT_SAPPHIRE / COLLECT_RUBY 成就）。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'COLLECT_COAL'},
                        {'type': 'achievement', 'name': 'COLLECT_IRON'},
                        {'type': 'achievement', 'name': 'COLLECT_DIAMOND'},
                        {'type': 'achievement', 'name': 'COLLECT_SAPPHIRE'},
                        {'type': 'achievement', 'name': 'COLLECT_RUBY'}]},
        "dependencies": ['native.collect_coal',
         'native.collect_iron',
         'native.collect_diamond',
         'native.collect_sapphire',
         'native.collect_ruby',
         'native.craft_diamond_pickaxe'],
    },
    {
        "task_id": 'native.collect_all_primary_resources',
        "instruction": 'Collect all primary resources. / 收集全部基础资源。',
        "objective": '同时收集木材、石头、煤炭、铁矿石与钻石（COLLECT_WOOD / COLLECT_STONE / COLLECT_COAL / COLLECT_IRON / COLLECT_DIAMOND 成就）。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'COLLECT_WOOD'},
                        {'type': 'achievement', 'name': 'COLLECT_STONE'},
                        {'type': 'achievement', 'name': 'COLLECT_COAL'},
                        {'type': 'achievement', 'name': 'COLLECT_IRON'},
                        {'type': 'achievement', 'name': 'COLLECT_DIAMOND'}]},
        "dependencies": ['native.collect_wood',
         'native.collect_stone',
         'native.collect_coal',
         'native.collect_iron',
         'native.collect_diamond',
         'native.craft_iron_pickaxe'],
    },
    {
        "task_id": 'native.conquer_dungeon_bosses',
        "instruction": 'Conquer the dungeon bosses. Defeat a troll, a knight, and the necromancer boss. / 征服地下城 Boss。击败巨魔、骑士与亡灵法师 Boss。',
        "objective": '同时达成 DEFEAT_TROLL、DEFEAT_KNIGHT 与 DEFEAT_NECROMANCER，击败后期最强敌人，作为 combat 类别的终极（root）目标。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'DEFEAT_TROLL'},
                        {'type': 'achievement', 'name': 'DEFEAT_KNIGHT'},
                        {'type': 'achievement', 'name': 'DEFEAT_NECROMANCER'}]},
        "dependencies": ['native.defeat_troll', 'native.defeat_knight', 'native.defeat_necromancer'],
    },
    {
        "task_id": 'native.conquer_lower_realms',
        "instruction": 'Conquer the lower realms: troll mines, fire realm and ice realm. / 征服下层领域：巨魔矿洞、火焰领域与寒冰领域。',
        "objective": '依次打通巨魔矿洞、火焰领域与寒冰领域（三者 ENTER_* 成就全部达成），直通墓地。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'ENTER_TROLL_MINES'},
                        {'type': 'achievement', 'name': 'ENTER_FIRE_REALM'},
                        {'type': 'achievement', 'name': 'ENTER_ICE_REALM'}]},
        "dependencies": ['native.enter_troll_mines', 'native.enter_fire_realm', 'native.enter_ice_realm'],
    },
    {
        "task_id": 'native.conquer_mid_tier_foes',
        "instruction": 'Conquer the mid-tier dungeon foes. Defeat a gnome warrior, a gnome archer, an orc soldier, an orc mage, and a kobold. / 征服中期地牢敌人。分别击败侏儒战士、侏儒弓箭手、兽人士兵、兽人法师和狗头人。',
        "objective": '同时达成 DEFEAT_GNOME_WARRIOR、DEFEAT_GNOME_ARCHER、DEFEAT_ORC_SOLIDER、DEFEAT_ORC_MAGE 与 DEFEAT_KOBOLD，覆盖 gnomish mines 与 sewers 两层的全部敌人，作为中期战斗阶段验收。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'DEFEAT_GNOME_WARRIOR'},
                        {'type': 'achievement', 'name': 'DEFEAT_GNOME_ARCHER'},
                        {'type': 'achievement', 'name': 'DEFEAT_ORC_SOLIDER'},
                        {'type': 'achievement', 'name': 'DEFEAT_ORC_MAGE'},
                        {'type': 'achievement', 'name': 'DEFEAT_KOBOLD'}]},
        "dependencies": ['native.defeat_gnome_warrior',
         'native.defeat_gnome_archer',
         'native.defeat_orc_soldier',
         'native.defeat_orc_mage',
         'native.defeat_kobold'],
    },
    {
        "task_id": 'native.crafting_mastery',
        "instruction": 'Master all crafting: collect all four pickaxes and a full kit. / 精通全部制作：集齐四种镐并打造一套完整装备。',
        "objective": '达成 master_crafter（四种镐全部成就）与 craft_full_kit（任意镐 + 任意剑 + 铁/钻石盔甲）的组合目标。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'and',
                         'predicates': [{'type': 'achievement', 'name': 'MAKE_WOOD_PICKAXE'},
                                        {'type': 'achievement', 'name': 'MAKE_STONE_PICKAXE'},
                                        {'type': 'achievement', 'name': 'MAKE_IRON_PICKAXE'},
                                        {'type': 'achievement',
                                         'name': 'MAKE_DIAMOND_PICKAXE'}]},
                        {'type': 'or',
                         'predicates': [{'type': 'achievement', 'name': 'MAKE_WOOD_PICKAXE'},
                                        {'type': 'achievement', 'name': 'MAKE_STONE_PICKAXE'},
                                        {'type': 'achievement', 'name': 'MAKE_IRON_PICKAXE'},
                                        {'type': 'achievement',
                                         'name': 'MAKE_DIAMOND_PICKAXE'}]},
                        {'type': 'or',
                         'predicates': [{'type': 'achievement', 'name': 'MAKE_WOOD_SWORD'},
                                        {'type': 'achievement', 'name': 'MAKE_STONE_SWORD'},
                                        {'type': 'achievement', 'name': 'MAKE_IRON_SWORD'},
                                        {'type': 'achievement', 'name': 'MAKE_DIAMOND_SWORD'}]},
                        {'type': 'or',
                         'predicates': [{'type': 'achievement', 'name': 'MAKE_IRON_ARMOUR'},
                                        {'type': 'achievement',
                                         'name': 'MAKE_DIAMOND_ARMOUR'}]}]},
        "dependencies": ['native.master_crafter', 'native.craft_full_kit'],
    },
    {
        "task_id": 'native.dungeon_campaign',
        "instruction": 'Complete the dungeon campaign. / 完成地下城战役。',
        "objective": '从收集木材、制作工具起步，一路战斗并下探到 Boss 层（player_level >= 8），完整走通本类别主线。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'COLLECT_WOOD'},
                        {'type': 'or',
                         'predicates': [{'type': 'achievement', 'name': 'MAKE_WOOD_PICKAXE'},
                                        {'type': 'achievement', 'name': 'MAKE_STONE_PICKAXE'},
                                        {'type': 'achievement', 'name': 'MAKE_IRON_PICKAXE'},
                                        {'type': 'achievement',
                                         'name': 'MAKE_DIAMOND_PICKAXE'}]},
                        {'type': 'or',
                         'predicates': [{'type': 'achievement', 'name': 'DEFEAT_ZOMBIE'},
                                        {'type': 'achievement', 'name': 'DEFEAT_SKELETON'},
                                        {'type': 'achievement', 'name': 'DEFEAT_GNOME_WARRIOR'},
                                        {'type': 'achievement', 'name': 'DEFEAT_GNOME_ARCHER'},
                                        {'type': 'achievement', 'name': 'DEFEAT_ORC_SOLIDER'},
                                        {'type': 'achievement', 'name': 'DEFEAT_ORC_MAGE'},
                                        {'type': 'achievement', 'name': 'DEFEAT_LIZARD'},
                                        {'type': 'achievement', 'name': 'DEFEAT_KOBOLD'},
                                        {'type': 'achievement', 'name': 'DEFEAT_TROLL'},
                                        {'type': 'achievement', 'name': 'DEFEAT_DEEP_THING'},
                                        {'type': 'achievement', 'name': 'DEFEAT_PIGMAN'},
                                        {'type': 'achievement',
                                         'name': 'DEFEAT_FIRE_ELEMENTAL'},
                                        {'type': 'achievement', 'name': 'DEFEAT_FROST_TROLL'},
                                        {'type': 'achievement', 'name': 'DEFEAT_ICE_ELEMENTAL'},
                                        {'type': 'achievement', 'name': 'DEFEAT_NECROMANCER'}]},
                        {'type': 'level_ge', 'value': 8}]},
        "dependencies": ['native.collect_wood',
         'native.craft_tools',
         'native.defeat_enemy',
         'native.explore_dungeon'],
    },
    {
        "task_id": 'native.iron_gear',
        "instruction": 'Craft a full iron gear set: iron pickaxe, iron sword and iron armour. / 打造一整套铁质装备：铁镐、铁剑与铁盔甲。',
        "objective": '同时拥有铁镐、铁剑和铁盔甲（MAKE_IRON_PICKAXE + MAKE_IRON_SWORD + MAKE_IRON_ARMOUR 成就全部达成）。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'MAKE_IRON_PICKAXE'},
                        {'type': 'achievement', 'name': 'MAKE_IRON_SWORD'},
                        {'type': 'achievement', 'name': 'MAKE_IRON_ARMOUR'}]},
        "dependencies": ['native.craft_iron_pickaxe', 'native.craft_iron_sword', 'native.craft_iron_armour'],
    },
    {
        "task_id": 'native.reach_mid_dungeon',
        "instruction": 'Reach the mid-dungeon: sewers and vault. / 推进到中层地牢：下水道与宝库。',
        "objective": '同时达成进入下水道与进入宝库两个成就，标志穿越地下城的中段里程碑。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'ENTER_SEWERS'},
                        {'type': 'achievement', 'name': 'ENTER_VAULT'}]},
        "dependencies": ['native.enter_sewers', 'native.enter_vault'],
    },
    {
        "task_id": 'native.starter_toolkit',
        "instruction": 'Craft a starter toolkit: wooden pickaxe, wooden sword, torch and arrow. / 打造新手工具包：木镐、木剑、火把与箭。',
        "objective": '同时拥有木镐、木剑、火把和箭（MAKE_WOOD_PICKAXE + MAKE_WOOD_SWORD + MAKE_TORCH + MAKE_ARROW 成就全部达成）。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'MAKE_WOOD_PICKAXE'},
                        {'type': 'achievement', 'name': 'MAKE_WOOD_SWORD'},
                        {'type': 'achievement', 'name': 'MAKE_TORCH'},
                        {'type': 'achievement', 'name': 'MAKE_ARROW'}]},
        "dependencies": ['native.craft_wood_pickaxe',
         'native.craft_wood_sword',
         'native.craft_torch',
         'native.craft_arrow'],
    },
    {
        "task_id": 'native.survival_starter_kit',
        "instruction": 'Gather wood and craft a basic tool. / 收集木材并制作一把基础工具。',
        "objective": '完成生存起步：收集木材并制作任意一种镐（木/石/铁/钻石镐任一），具备采矿与工具制作能力。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'COLLECT_WOOD'},
                        {'type': 'or',
                         'predicates': [{'type': 'achievement', 'name': 'MAKE_WOOD_PICKAXE'},
                                        {'type': 'achievement', 'name': 'MAKE_STONE_PICKAXE'},
                                        {'type': 'achievement', 'name': 'MAKE_IRON_PICKAXE'},
                                        {'type': 'achievement',
                                         'name': 'MAKE_DIAMOND_PICKAXE'}]}]},
        "dependencies": ['native.collect_wood', 'native.craft_tools'],
    },
    {
        "task_id": 'native.survival_sustenance',
        "instruction": 'Secure food and drink. / 解决食物与饮水，维持生存。',
        "objective": '获得饮用水（COLLECT_DRINK）并食用任意一种食物（EAT_COW / EAT_PLANT / EAT_BAT / EAT_SNAIL 任一）。',
        "success_predicate": {'type': 'and',
         'predicates': [{'type': 'achievement', 'name': 'COLLECT_DRINK'},
                        {'type': 'or',
                         'predicates': [{'type': 'achievement', 'name': 'EAT_COW'},
                                        {'type': 'achievement', 'name': 'EAT_PLANT'},
                                        {'type': 'achievement', 'name': 'EAT_BAT'},
                                        {'type': 'achievement', 'name': 'EAT_SNAIL'}]}]},
        "dependencies": ['native.collect_drink', 'native.eat_food'],
    },
]


def register_hierarchy_tasks() -> None:
    from craftax.tasks.registry import register

    for d in _HIERARCHY_DEFS:
        achievements = _collect_achievements(d['success_predicate'])
        spec = TaskSpec(
            task_id=d['task_id'],
            version=TASK_VERSION,
            instruction=d['instruction'],
            objective=d['objective'],
            success_predicate=d['success_predicate'],
            annotation_predicates=[{'type': 'achievement', 'name': a} for a in achievements],
            renderer_config={},
            dependencies=d['dependencies'],
        )
        register(spec.task_id, spec.version, lambda spec=spec: BaseTaskAdapter(spec))


def _collect_achievements(expr: Any) -> List[str]:
    """递归收集谓词中引用的全部成就名（保持顺序、去重）。"""
    out: List[str] = []

    def walk(e: Any) -> None:
        if isinstance(e, dict):
            if str(e.get('type', '')) == 'achievement':
                name = str(e.get('name', ''))
                if name and name not in out:
                    out.append(name)
            for value in e.values():
                walk(value)
        elif isinstance(e, (list, tuple)):
            for item in e:
                walk(item)

    walk(expr)
    return out


register_hierarchy_tasks()
