import json
import os   
import re
import sys
from typing import Generator, List, Tuple, Optional

from og_ego_prim.models.hf_inference import HFClient
from og_ego_prim.models.server_inference import ServerClient
from og_ego_prim.models.base_client import BaseClient
from og_ego_prim.primitives import VALID_PRIMITIVES
from og_ego_prim.benchmark.tracker import EvalTracker
from og_ego_prim.utils.constants import WORK_DIR
from og_ego_prim.utils.prompts import *
from og_ego_prim.utils.types import StepwisePlan

from og_ego_prim.utils.constants import TASKS

class BadAgentPlanError(Exception):
    pass


def parse_output(output: str) -> Optional[StepwisePlan]:
    pattern = r'```json(.*?)```'
    result = re.findall(pattern, output, re.DOTALL)

    if len(result) >= 1:
        result = result[0].strip()
        try:
            result = json.loads(result)
        except:
            result = None
        return result
    # Fallback: raw JSON (no fence) — SFT-trained model emits this form
    text = output.strip()
    if text.startswith('{') and text.endswith('}'):
        try:
            return json.loads(text)
        except:
            pass
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return json.loads(text[start:end])
    except:
        return None


def parse_candidates_output(output: str) -> Optional[List[StepwisePlan]]:
    """V0/V1/V4CandidatesPrompt 응답에서 JSON 배열을 파싱."""
    pattern = r'```json(.*?)```'
    result = re.findall(pattern, output, re.DOTALL)
    if result:
        try:
            parsed = json.loads(result[0].strip())
            if isinstance(parsed, list):
                return parsed
        except:
            pass
    # fallback: 배열 형태 직접 파싱
    try:
        start = output.index('[')
        end = output.rindex(']') + 1
        parsed = json.loads(output[start:end])
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    return None


def get_obs_from_dir(obs_dir: str) -> List[str]:
    obs_path_list = []
    for img in sorted(os.listdir(obs_dir)):
        img_path = os.path.join(obs_dir, img)
        if img.endswith(".png"):
            obs_path_list.append(img_path)
    return obs_path_list


class PlanningAgent: 
    
    def __init__(
        self, 
        task_name: str, 
        scene_name: str, 
        agent_name: str, 
        work_dir: str,
        local_llm_serve: str,
        local_serve_ip: str,
        local_serve_key: str,
        prompt_setting: str,
        use_initial_setup: bool = False,
        use_self_caption: bool = False,
        retry: int = 3,
        verbose: bool = True,
        debug: bool = False,
    ) -> None:
        if work_dir is None:
            work_dir = WORK_DIR
        self.working_dir = os.path.join(work_dir, "benchmark")
        assert os.path.exists(self.working_dir)

        self.task_name = task_name
        self.scene_name = scene_name
        self.agent_name = agent_name
        self.current_step = 0

        self.retry = retry
        self.verbose = verbose
        self.debug = debug
        
        self.local_llm_serve = local_llm_serve
        self.local_serve_ip = local_serve_ip
        self.local_serve_key = local_serve_key
        self.prompt_setting = prompt_setting
        self.use_initial_setup = use_initial_setup
        self.use_self_caption = use_self_caption

        # KB-injected safety guidelines (for v4 prompt)
        self.kb_guidelines: Optional[str] = None

        # initialize data
        (self.task_instruction, self.objects_str, self.initial_setup_str,
         self.object_abilities_str, self.wash_rules_str, self.goal_bddl_str,
         self.safety_tips_str, self.rich_safety_tips_str) = self.load_info_data()
        if self.verbose:
            print(f'[agent] instruction: {self.task_instruction}')
            print(f'[agent] objects:\n{self.objects_str}')
            print(f'[agent] initial setup:\n{self.initial_setup_str}')
            print(f'[agent] object abilities:\n{self.object_abilities_str}')
            print(f'[agent] wash rules:\n{self.wash_rules_str}')
            print(f'[agent] goal bddl:\n{self.goal_bddl_str}')
            sys.stdout.flush()
        
        self.client = self._get_agent(agent_name)
    
    def set_tracker(self, tracker: EvalTracker):
        self.tracker = tracker
        model_name = self.agent_name.split("/")[-1]
        self.tracker.model = model_name

    def set_kb_guidelines(self, guidelines_str: str) -> None:
        """Set KB-injected safety guidelines for v4 prompt."""
        self.kb_guidelines = guidelines_str

    def _get_agent(self, agent_name: str) -> BaseClient:
        if self.local_llm_serve: 
            return ServerClient(
                model_type="local", 
                model_name=agent_name,
                api_key=self.local_serve_key, 
                api_base=self.local_serve_ip
            ) 
        else: 
            return ServerClient(
                model_type="close_source",
                model_name=agent_name, 
                api_key=os.environ['OPENAI_API_KEY'], 
                api_base=os.environ.get('OPENAI_API_BASE') or None
            ) 

    def _get_last_execution_info(self, use_obs=True):
        last_step, last_plan = 0, 'init'
        for plan in reversed(self.tracker.plans):
            if not plan['plan']['action'].startswith('navigate'):
                last_step = plan['step']
                last_plan = plan['plan']['action']
                break
        
        if not use_obs:
            observations = None
        else:
            benchmark_tag = f'{self.task_name}___{self.scene_name}'
            model_tag = self.agent_name.replace('/', '__')
            step_tag = f'{last_step}_' + last_plan.replace('(', '__').replace(')', '__')
            obs_dir = os.path.join(self.working_dir, benchmark_tag, model_tag, step_tag)
            observations = get_obs_from_dir(obs_dir)

            print(f'read obs from {obs_dir}')
            sys.stdout.flush()
        
        return last_plan, observations

    def _prepare_prompt(self) -> str:
        history_plans = "None"
        if self.current_step > 0:
            history_plans = '\n'.join(
                [history['history_text'] for history in self.tracker.plans]
            )
            
        if not self.use_initial_setup and not self.use_self_caption:
            if self.prompt_setting == 'v0': # v0: no safety reminder
                prompt = V0StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans
                )
            elif self.prompt_setting == 'v1': # v0 + implicit safety reminder
                prompt = V1StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans
                )
            elif self.prompt_setting == 'v2': # v0 + cot safety reminder
                assert self.tracker.awareness is not None and 'content' in self.tracker.awareness
                awareness = self.tracker.awareness['content']
                prompt = V2StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    awareness=awareness
                )
            elif self.prompt_setting == 'v3': # v0 + explicit safety reminder
                prompt = V3StepPlanningPrompt.format(
                    objects_str=self.objects_str,
                    task_instruction=self.task_instruction,
                    object_abilities_str=self.object_abilities_str,
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    safety_tips=self.safety_tips_str,
                    history_actions=history_plans
                )
            else:
                raise Exception('Wrong prompt setting.')
        else:
            if self.use_initial_setup:
                scene_description = self.initial_setup_str
            else:
                assert self.tracker.caption is not None and 'content' in self.tracker.caption
                scene_description = self.tracker.caption['content']

            if self.prompt_setting == 'v0': 
                prompt = T0StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    scene_description=scene_description, 
                )
            elif self.prompt_setting == 'v1': 
                prompt = T1StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    scene_description=scene_description, 
                )
            elif self.prompt_setting == 'v2':
                assert self.tracker.awareness is not None and 'content' in self.tracker.awareness
                awareness = self.tracker.awareness['content'] 
                prompt = T2StepPlanningPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    scene_description=scene_description, 
                    awareness=awareness
                )
            elif self.prompt_setting == 'v3':
                prompt = T3StepPlanningPrompt.format(
                    objects_str=self.objects_str,
                    task_instruction=self.task_instruction,
                    object_abilities_str=self.object_abilities_str,
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    history_actions=history_plans,
                    safety_tips=self.safety_tips_str,
                    scene_description=scene_description
                )
            elif self.prompt_setting == 'v4':
                # v4 uses V4StepPlanningPrompt (no T4 variant; uses V4 without scene_description)
                prompt = V4StepPlanningPrompt.format(
                    objects_str=self.objects_str,
                    task_instruction=self.task_instruction,
                    object_abilities_str=self.object_abilities_str,
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                    kb_safety_guidelines=self.kb_guidelines or "None",
                    history_actions=history_plans
                )
            else:
                raise Exception('Wrong prompt setting.')

        return prompt
    

    def _verify_plan(self, plan: Optional[StepwisePlan]) -> Optional[Tuple[str, str, str]]:
        if plan is None:
            return None
        if 'action' not in plan:
            return None
        
        action = plan['action'].strip()
        if action.upper().startswith('DONE'):
            caution = plan.get('caution', None)
            return 'done', '', caution

        pattern = r'(?:\d+\.\s+)?([a-zA-Z_]+)\(([^)]+)\)'
        matches = re.findall(pattern, action)
        if len(matches) >= 1:
            operator, params = matches[-1]
        else:
            return None
        
        operator = operator.strip()
        if operator.upper() not in VALID_PRIMITIVES.keys():
            return None

        params = params.strip().lower()
        objects = [obj.strip() for obj in params.split(',')]
        if len(objects) != VALID_PRIMITIVES[operator.upper()]:
            return None
        for obj in objects:
            if obj not in self.objects_str:
                return None

        if 'caution' not in plan:
            caution = None
        else:
            caution = plan['caution']
        return operator.lower(), params, caution
    
    def _prepare_candidates_prompt(self, k: int) -> str:
        """후보 생성용 프롬프트. prompt_setting에 따라 v0/v1/v4 분기."""
        history_plans = "None"
        if self.current_step > 0:
            history_plans = '\n'.join(
                [history['history_text'] for history in self.tracker.plans]
            )
        common_kwargs = dict(
            task_instruction=self.task_instruction,
            objects_str=self.objects_str,
            object_abilities_str=self.object_abilities_str,
            task_goals=self.goal_bddl_str,
            wash_rules_str=self.wash_rules_str,
            history_actions=history_plans,
            k=k,
        )
        if self.prompt_setting == 'v4' and self.kb_guidelines:
            return V4CandidatesPrompt.format(
                kb_safety_guidelines=self.kb_guidelines,
                **common_kwargs,
            )
        if self.prompt_setting == 'v0':
            return V0CandidatesPrompt.format(**common_kwargs)
        return V1CandidatesPrompt.format(**common_kwargs)

    def generate_candidates(
        self,
        k: int = 3,
        image_file=None,
        temperature: float = 1.0,
        single_call: bool = False,
    ) -> List[StepwisePlan]:
        """
        VLM에게 k개 후보 action을 요청.

        single_call=False (기본, MCTS용):
            k=1짜리 프롬프트를 최대 k*retry번 독립 호출 → temperature 샘플링으로 다양성 확보.

        single_call=True (Beam Search용):
            k개를 한번에 요청 → VLM이 한 컨텍스트에서 k가지 다른 전략을 명시적으로 제안.
            이미지 관찰도 한번만 수행하므로 더 일관된 state awareness 보장.
            실패 시 최대 retry번 재시도.
        """
        gen_args = {"max_completion_tokens": 1024, "temperature": temperature}
        seen_actions: set = set()
        valid: List[StepwisePlan] = []

        if single_call:
            # Single call: ask for k diverse candidates at once
            prompt = self._prepare_candidates_prompt(k=k)
            for _ in range(self.retry):
                output = self.client.model(prompt, image_file=image_file, gen_args=gen_args)
                if self.verbose:
                    print(f"[agent] candidate raw output:\n{output}")
                candidates_raw = parse_candidates_output(output)
                if not candidates_raw:
                    continue
                for raw in candidates_raw:
                    result = self._verify_plan(raw)
                    if result is None:
                        continue
                    operator, params, caution = result
                    action = f'{operator}({params})'
                    if action not in seen_actions:
                        seen_actions.add(action)
                        valid.append(StepwisePlan(action=action, caution=caution))
                if valid:
                    break
        else:
            # Independent calls: k=1 prompt called up to k*retry times
            prompt = self._prepare_candidates_prompt(k=1)
            gen_args["max_completion_tokens"] = 512
            for _ in range(k * self.retry):
                if len(valid) >= k:
                    break
                output = self.client.model(prompt, image_file=image_file, gen_args=gen_args)
                if self.verbose:
                    print(f"[agent] candidate raw output:\n{output}")
                candidates_raw = parse_candidates_output(output)
                if not candidates_raw:
                    continue
                result = self._verify_plan(candidates_raw[0])
                if result is None:
                    continue
                operator, params, caution = result
                action = f'{operator}({params})'
                if action not in seen_actions:
                    seen_actions.add(action)
                    valid.append(StepwisePlan(action=action, caution=caution))

        return valid

    def generate_caption(self, use_obs=True) -> str:
        _, obs = self._get_last_execution_info(use_obs)
        prompt_cp = GenerateCaptionPrompt.format(
                objects_str=self.objects_str, 
                task_instruction=self.task_instruction, 
                object_abilities_str=self.object_abilities_str, 
                task_goals=self.goal_bddl_str,
                wash_rules_str=self.wash_rules_str,
            )
        output_caption = self.client.model(prompt_cp, image_file=obs)
        return output_caption
        
    def generate_awareness(self, use_obs=True) -> str:
        _, obs = self._get_last_execution_info(use_obs)
        if self.use_initial_setup or self.use_self_caption: 
            if self.use_initial_setup:
                scene_description = self.initial_setup_str
            else:
                assert self.tracker.caption is not None and 'content' in self.tracker.caption
                scene_description = self.tracker.caption['content']
            prompt_sa = T2GenerateAwarenessPrompt.format(
                objects_str=self.objects_str, 
                task_instruction=self.task_instruction, 
                object_abilities_str=self.object_abilities_str, 
                task_goals=self.goal_bddl_str,
                wash_rules_str=self.wash_rules_str,
                scene_description=scene_description, 
            )
        else:
            prompt_sa = GenerateAwarenessPrompt.format(
                    objects_str=self.objects_str, 
                    task_instruction=self.task_instruction, 
                    object_abilities_str=self.object_abilities_str, 
                    task_goals=self.goal_bddl_str,
                    wash_rules_str=self.wash_rules_str,
                )
        output = self.client.model(prompt_sa, image_file=obs)
        return output
        
        

    def step(self, use_obs=True, max_step=None) -> Generator[str, None, None]:
        retry = 0
        while True:
            # get obs after last execution
            last_plan, obs = self._get_last_execution_info(use_obs)
            prompt = self._prepare_prompt()

            if self.debug:
                print(f'[agent] last_step: {last_plan}, Continue (y/Y): ')
                sys.stdout.flush()

                while cmd := input().upper() != 'Y':
                    print(f'[agent] last_step: {last_plan}, Continue (y/Y): ')
                    sys.stdout.flush()
            
            output = self.client.model(prompt, image_file=obs)
            next_plan = parse_output(output)
            if self.verbose:
                print(f"[agent] raw output:\n{output}")
                print(f"[agent] next plan:\n{next_plan}")
                sys.stdout.flush()

            # verification the next step of generated plan is correct
            results = self._verify_plan(next_plan)
            if results is None:
                retry += 1
                if retry < self.retry:
                    print(f"[agent] retry...")
                    sys.stdout.flush()
                    continue
                else:
                    self.tracker.track_termination(
                        reason='plan_error',
                        type='BadAgentPlanError',
                        msg=f'plan ``{next_plan if next_plan else "None"}`` not applicable'
                    )
                    return
            else:
                retry = 0
            
                operator, params, caution = results
                self.current_step += 1
                next_plan: StepwisePlan = dict(
                    action=f'{operator}({params})',
                    caution=caution
                )
                self.tracker.track_plan(
                    step=self.current_step,
                    plan=next_plan,
                    history_text=f'{self.current_step}. {operator.upper()}({params.lower()})'
                )
                self.tracker.track_raw_output(
                    step=self.current_step,
                    content=output,
                )

                yield next_plan
                if operator == 'done':
                    return
                if max_step is not None and self.current_step > max_step:
                    self.tracker.track_termination(
                        reason='exceeding_max_steps',
                        type='BadAgentPlanError',
                        msg=f'exceeding max steps {max_step}'
                    )
                    return
        
    @staticmethod
    def _format_rich_safety_tips(task_json_data: dict) -> str:
        return ""

    @staticmethod
    def _format_wash_rules(wash_rules: list) -> str:
        """Convert wash_rules JSON into human-readable, action-oriented text.

        Translates the condition structure into concrete primitive suggestions so
        the agent does not have to infer which SOAK variant to use.

        Example input:
            [{"cleaning_tool": "sponge", "support_system": "stain",
              "conditions": [{"type": "saturated", "value": "liquid_soap",
                              "source": ["liquid_soap__bottle.n.01_1"]}]}]
        Example output:
            "Wash rule 1: To remove stain — use SOAK_INSIDE(sponge, liquid_soap__bottle.n.01_1)
             to saturate sponge with liquid_soap, then WIPE(surface, sponge)."
        """
        lines = []
        for idx, rule in enumerate(wash_rules, 1):
            tool = rule.get('cleaning_tool', 'cleaning_tool')
            support = rule.get('support_system', 'dirt')
            conditions = rule.get('conditions') or []

            if not conditions:
                # If the rule does not specify preparation conditions, avoid
                # inventing extra sink/soaking steps and describe direct wiping.
                lines.append(
                    f"Wash rule {idx}: To remove {support} — "
                    f"use WIPE(surface, {tool})."
                )
                continue

            prep_steps = []
            for cond in conditions:
                ctype = cond.get('type', '')
                value = cond.get('value', '')
                sources = cond.get('source') or []
                source = sources[0] if sources else None

                if ctype == 'saturated' and source:
                    # Need to immerse tool in a liquid container
                    if 'sink' in (source or '').lower():
                        prep_steps.append(
                            f"SOAK_UNDER({tool}, {source}) to saturate {tool} with {value}"
                        )
                    else:
                        prep_steps.append(
                            f"SOAK_INSIDE({tool}, {source}) to saturate {tool} with {value}"
                        )
                elif ctype in ('wet', 'soaked') and source:
                    if 'sink' in (source or '').lower():
                        prep_steps.append(f"SOAK_UNDER({tool}, {source}) to wet {tool}")
                    else:
                        prep_steps.append(f"SOAK_INSIDE({tool}, {source}) to wet {tool}")
                elif ctype == 'toggled_on' and source:
                    prep_steps.append(f"TOGGLE_ON({source}) before wiping")
                else:
                    # Unknown condition — include raw for transparency
                    prep_steps.append(f"satisfy condition: {json.dumps(cond)}")

            prep_str = '; then '.join(prep_steps)
            lines.append(
                f"Wash rule {idx}: To remove {support} — "
                f"{prep_str}, then WIPE(surface, {tool})."
            )

        return '\n'.join(lines)

    def load_info_data(self):
        with open(os.path.join(TASKS, f"{self.task_name}.json"), 'r', encoding='utf-8') as f:
            task_json_data = json.load(f)
        task_instruction = task_json_data['planning_context']['task_instruction']
        objects_list = task_json_data['planning_context']['object_list']
        objects_str = '\n'.join(f"{i+1}. {item.strip()}" for i, item in enumerate(objects_list))
        intial_setup_list = task_json_data['planning_context']['initial_setup']
        initial_setup_str = '\n'.join(f"{item.strip()}" for i, item in enumerate(intial_setup_list))
        
        object_abilities = task_json_data['planning_context']['object_abilities']
        if object_abilities is None:
            object_abilities_str = ""
        else:
            object_abilities_str = '\n'.join([f"{key}: " + str(value) for key, value in object_abilities.items()])
            
        wash_rules = task_json_data['planning_context']['wash_rules']
        if wash_rules is None:
            wash_rules_str = ""
        else:
            wash_rules_str = self._format_wash_rules(wash_rules)

        safety_tips = []
        for tip in task_json_data['evaluation_cautions']:
            safety_tips.append(tip['safety_tip'])
        for tip in task_json_data['evaluation_goal_conditions']['process_safety_goal_condition']:
            safety_tips.append(tip['safety_tip'])
        for tip in task_json_data['evaluation_goal_conditions']['termination_safety_goal_condition']:
            safety_tips.append(tip['safety_tip'])
        safety_tips_str = json.dumps(safety_tips, indent=4, ensure_ascii=False)

        rich_safety_tips_str = self._format_rich_safety_tips(task_json_data)

        goal_condition_bddl_str = task_json_data['evaluation_goal_conditions']['execution_goal_condition']

        return task_instruction, objects_str, initial_setup_str, object_abilities_str, wash_rules_str, goal_condition_bddl_str, safety_tips_str, rich_safety_tips_str
