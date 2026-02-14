#!/usr/bin/env python3

# ---------------------------
# Install libraries
# ---------------------------
# !pip install tqdm
# !pip install vllm
# !pip install numpy==1.25.0
# !huggingface-cli login --token YOUR_HF_TOKEN


# ---------------------------
# Imports and Setup
# ---------------------------
import os
import random
import json
import time
import pandas as pd
from tqdm import tqdm
from json import JSONDecodeError
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ------------------------------------------------------------------
# 1. Dictionaries & Settings
# ------------------------------------------------------------------

languages = {
    "hat": {"name": "Haitian Creole", "category": "tail"},
    "jam": {"name": "Jamaican Patois", "category": "tail"},
    "amh": {"name": "Amharic", "category": "tail"},
    "ibo": {"name": "Igbo", "category": "tail"},
    "ful": {"name": "Fulani", "category": "tail"},
    "zul": {"name": "Zulu", "category": "tail"},
    "pap": {"name": "Papiamento", "category": "tail"},
    "afr": {"name": "Afrikaans", "category": "tail"},
    "afr": {"name": "Afrikaans", "category": "tail"},
    "ara": {"name": "Arabic", "category": "head"},
    "aze": {"name": "Azerbaijani", "category": "tail"},
    "ban": {"name": "Balinese", "category": "tail"},
    "ben": {"name": "Bengali", "category": "tail"},
    "bos": {"name": "Bosnian", "category": "tail"},
    "bul": {"name": "Bulgarian", "category": "tail"},
    "cat": {"name": "Catalan", "category": "tail"},
    "ces": {"name": "Czech", "category": "head"},
    "dan": {"name": "Danish", "category": "head"},
    "deu": {"name": "German", "category": "head"},
    "ell": {"name": "Greek", "category": "head"},
    "eng": {"name": "English", "category": "head"},
    "est": {"name": "Estonian", "category": "tail"},
    "fas": {"name": "Persian", "category": "tail"},
    "fin": {"name": "Finnish", "category": "tail"},
    "fra": {"name": "French", "category": "head"},
    "grn": {"name": "Guarani", "category": "tail"},
    "guj": {"name": "Gujarati", "category": "tail"},
    "hau": {"name": "Hausa", "category": "tail"},
    "heb": {"name": "Hebrew", "category": "tail"},
    "hin": {"name": "Hindi", "category": "tail"},
    "hrv": {"name": "Croatian", "category": "tail"},
    "hun": {"name": "Hungarian", "category": "tail"},
    "ind": {"name": "Indonesian", "category": "head"},
    "ita": {"name": "Italian", "category": "head"},
    "jpn": {"name": "Japanese", "category": "head"},
    "kat": {"name": "Georgian", "category": "tail"},
    "kor": {"name": "Korean", "category": "head"},
    "kur": {"name": "Kurdish", "category": "tail"},
    "lav": {"name": "Latvian", "category": "tail"},
    "lit": {"name": "Lithuanian", "category": "tail"},
    "mal": {"name": "Malayalam", "category": "tail"},
    "mar": {"name": "Marathi", "category": "tail"},
    "mkd": {"name": "Macedonian", "category": "tail"},
    "msa": {"name": "Malay", "category": "tail"},
    "mya": {"name": "Burmese", "category": "tail"},
    "nep": {"name": "Nepali", "category": "tail"},
    "nld": {"name": "Dutch", "category": "head"},
    "nor": {"name": "Norwegian", "category": "tail"},
    "orm": {"name": "Oromo", "category": "tail"},
    "pan": {"name": "Punjabi", "category": "tail"},
    "per": {"name": "Persian", "category": "tail"},
    "pol": {"name": "Polish", "category": "head"},
    "por": {"name": "Portuguese", "category": "head"},
    "ron": {"name": "Romanian", "category": "tail"},
    "rus": {"name": "Russian", "category": "head"},
    "sin": {"name": "Sinhala", "category": "tail"},
    "slk": {"name": "Slovak", "category": "tail"},
    "som": {"name": "Somali", "category": "tail"},
    "spa": {"name": "Spanish", "category": "head"},
    "sqi": {"name": "Albanian", "category": "tail"},
    "srp": {"name": "Serbian", "category": "tail"},
    "swa": {"name": "Swahili", "category": "tail"},
    "swe": {"name": "Swedish", "category": "tail"},
    "tam": {"name": "Tamil", "category": "tail"},
    "tel": {"name": "Telugu", "category": "tail"},
    "tgl": {"name": "Tagalog", "category": "tail"},
    "tha": {"name": "Thai", "category": "tail"},
    "tur": {"name": "Turkish", "category": "head"},
    "ukr": {"name": "Ukrainian", "category": "head"},
    "urd": {"name": "Urdu", "category": "tail"},
    "vie": {"name": "Vietnamese", "category": "head"},
    "zho": {"name": "Chinese", "category": "head"}
}

real_news_targets = {
    "amh": 375, "ibo": 375, "ful": 375, "zul": 375, "pap": 375,  # New languages
    "hat": 252, "jam": 240, "ban": 177, "grn": 160, "urd": 153,
    "ukr": 150, "tha": 150, "orm": 149, "tgl": 148, "heb": 147,
    "zho": 147, "hrv": 146, "tur": 144, "vie": 144, "msa": 142,
    "per": 139, "nor": 138, "est": 137, "fin": 134, "eng": 133,
    "som": 125, "mya": 125, "mkd": 124, "pan": 123, "swe": 123,
    "mar": 123, "srp": 123, "tam": 122, "tel": 121, "nep": 118,
    "bos": 118, "swa": 116, "mal": 114, "kur": 114, "hau": 113,
    "lav": 112, "sin": 111, "bul": 111, "slk": 110, "spa": 104,
    "pol": 104, "ron": 104, "nld": 103, "ces": 103, "sqi": 102,
    "guj": 102, "jpn": 102, "lit": 101, "rus": 100, "kat": 99,
    "dan": 97, "kor": 96, "ben": 96, "por": 96, "hun": 96,
    "hin": 95, "aze": 95, "ita": 92, "fas": 89, "ara": 88,
    "cat": 88, "ind": 87, "afr": 87, "fra": 83, "ell": 73,
    "deu": 63
}

# ------------------------------------------------------------------
# 2. Techniques & Degrees
# ------------------------------------------------------------------

degree = {
    "light": "light change (10-20%) changes",
    "moderate": "moderate change (30-50%) changes",
    "complete": "complete change (100%)"
}

technique_placeholder = {
    "rewrite": {
        "technique_info": "rewriting, significantly restructuring and rephrasing the original content",
        "chain_placeholder": "Rewrite Humanizer",
        "role_placeholder": "You are a Rewriter and Humanizer specializing in comprehensive paraphrasing and natural language refinement.",
        "task_placeholder": (
            "Use the analysis from Chain [1] to rephrase and restructure significantly the original content, "
            "altering wording and sentence structures while maintaining complete factual accuracy. "
            "Apply {degree}. Then, humanize the rewritten text by refining it to exhibit natural language patterns."
        )
    },
    "polish": {
        "technique_info": "polishing the original content, refining language clarity and style",
        "chain_placeholder": "Polisher",
        "role_placeholder": "You are a Polisher specializing in refining language and stylistic presentation.",
        "task_placeholder": (
            "Polish the original content, refining clarity, flow, and readability without significantly "
            "altering the structure or factual content."
        )
    },
    "edit": {
        "technique_info": "editing the original content with minor adjustments, correcting grammar and small errors",
        "chain_placeholder": "Editor",
        "role_placeholder": "You are an Editor specializing in precise word-level edits and subtle content adjustments.",
        "task_placeholder": (
            "Perform minor content editing of the original text to improve quality, correct inaccuracies, "
            "and enhance readability."
        )
    }
}

techniques = ["rewrite", "polish", "edit"]
samples_per_technique = 80 // len(techniques)  # e.g. if total desired per language is 80

# ------------------------------------------------------------------
# 3. Output dirs & global UUID tracking
# ------------------------------------------------------------------

main_output_dir = "D:\\xGEN\\eng_xLang\\eng_xLang\\real_news\\vllm"
os.makedirs(main_output_dir, exist_ok=True)

global_used_uuids_file = os.path.join(main_output_dir, "global_used_uuids.json")
if os.path.exists(global_used_uuids_file):
    with open(global_used_uuids_file, 'r', encoding='utf-8') as f:
        global_used_uuids = set(json.load(f))
else:
    global_used_uuids = set()
    with open(global_used_uuids_file, 'w', encoding='utf-8') as f:
        json.dump(list(global_used_uuids), f, indent=2)

model_list = [
        # "openai/gpt-oss-20b",
        "Qwen/Qwen3-32B",
        "openai/gpt-oss-120b",
    ]

# ------------------------------------------------------------------
# 4. Point at your per-language folders (one CSV each)
# ------------------------------------------------------------------

input_data_dir = "original_data/real_news" # replace with your path

# ------------------------------------------------------------------
# 5. Helper functions
# ------------------------------------------------------------------

def update_overall_counts(path, success, fail):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'overall_success_count': success, 'overall_fail_count': fail}, f, indent=2)

def update_impersonator(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def build_prompt(placeholder, article, lang_name):
    """Constructs the JSON-based Chain-of-Interaction prompt."""
    task = placeholder["task_placeholder"]
    if "{degree}" in task and placeholder.get("degree"):
        task = task.replace("{degree}", degree[placeholder["degree"]])
    return f"""
            {{
              "GREEN_CoIAgenticF3": {{
            "Agentic_CoI": [
                  {{
                    "chain": "Chain [1] – Analyst/Examiner",
                    "role": "You are an Analyst/Examiner specializing in accurate content analysis.",
                    "task": "Analyze the provided article, extracting key ideas, facts, entities, sentiments, and biases or predispositions."
                  }},
                  {{
                    "chain": "Chain [2] – {placeholder['chain_placeholder']}",
                    "role": "{placeholder['role_placeholder']}",
                    "task": "{task}"
                  }},
                  {{
                    "chain": "Chain [3] – Validator/Quality Checker",
                    "role": "You are a Validator/Quality Checker specializing in accuracy verification.",
                    "task": "Validate the modified content to ensure all facts are accurately represented and note any discrepancies in English."
                  }},
                  {{
                    "chain": "Chain [4] – Adjuster/Fixer",
                    "role": "You are an Adjuster/Fixer specializing in applying corrections.",
                    "task": "Apply corrections based on validation to ensure the final content is coherent and factually accurate in {lang_name}."
                  }},
                  {{
                    "chain": "Chain [5] – Translator",
                    "role": "You are a Translator specializing in culturally sensitive translations.",
                    "task": "Translate the final corrected content into English, preserving accuracy and tone."
                  }},
                  {{
                    "chain": "Chain [6] – Localization QA/Reviewer",
                    "role": "You are a Localization QA/Reviewer specializing in cultural nuance and fluency.",
                    "task": "Review the English translation to ensure fluency, accuracy, and cultural appropriateness. Correct any issues."
                  }},
                  {{
                    "chain": "Chain [7] – Evaluator/Explainability Agent",
                    "role": "You are an Evaluator/Explainability Agent specializing in detailed assessments.",
                    "task": "Evaluate the final translated text on Accuracy, Fluency, Readability, and Naturalness using a 5-point Likert scale with justifications."
                  }},
                  {{
                    "chain": "Chain [8] – Output Formatter",
                    "role": "You are an Output Formatter specializing in concise social media posts.",
                    "task": "Produce two engaging social media posts (one in English and one in {lang_name}) summarizing key elements of the article using informal language and relevant hashtags."
                  }}
                ]
              }},
              "ChainOutputs": [
                {{
                  "Chain [1]": {{
                    "role": "Analyst/Examiner",
                    "analysis": {{
                      "key_ideas": [],
                      "facts_entities": [],
                      "sentiments": [],
                      "notable_biases": []
                    }}
                  }}
                }},
                {{
                  "Chain [2]": {{
                    "role": "{placeholder['chain_placeholder']}",
                    "modified_content": []
                  }}
                }},
                {{
                  "Chain [3]": {{
                    "role": "Validator/Quality Checker",
                    "validation_log": []
                  }}
                }},
                {{
                  "Chain [4]": {{
                    "role": "Adjuster/Fixer",
                    "final_corrected_content": []
                  }}
                }},
                {{
                  "Chain [5]": {{
                    "role": "Translator",
                    "translated_content": []
                  }}
                }},
                {{
                  "Chain [6]": {{
                    "role": "Localization QA/Reviewer",
                    "reviewed_translation": []
                  }}
                }},
                {{
                  "Chain [7]": {{
                    "role": "Evaluator/Explainability Agent",
                    "evaluation": {{
                      "Accuracy": {{
                        "score": "",
                        "justification": ""
                      }},
                      "Fluency": {{
                        "score": "",
                        "justification": ""
                      }},
                      "Readability": {{
                        "score": "",
                        "justification": ""
                      }},
                      "Naturalness": {{
                        "score": "",
                        "justification": ""
                      }}
                    }}
                  }}
                }},
                {{
                  "Chain [8]": {{
                    "role": "Output Formatter",
                    "English_output": "",
                    "{lang_name}_output": ""
                  }}
                }}
              ]
            }}
            Input News Article: {article}
            """

def create_new_persona(new_idx, impersonator, tokenizer, llm, params):
    """Auto-generate a new persona when all existing ones have refused."""
    data = json.dumps(impersonator, indent=2)
    system = "You are an ethical journalism mentor tasked with designing concise impersonation prompts that impose a positive role and intent."
    user_prompt = f"Generate ONE clever impersonation prompt based on these personas:\n{data}"
    chat = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    out = next(llm.generate(text, sampling_params=params)).outputs[0].text
    return out


# ------------------------------------------------------------------
# Prune global_used_uuids to match actually present files
# ------------------------------------------------------------------
def prune_global_uuids():
    present = set()
    for model_name in model_list:
        model_dir = os.path.join(main_output_dir, model_name.replace("/", "_"))
        for lc in languages:
            lang_dir = os.path.join(model_dir, lc)
            if not os.path.isdir(lang_dir):
                continue
            for fn in os.listdir(lang_dir):
                if fn.startswith("CoI_") and fn.endswith(".json"):
                    parts = fn.split("_")
                    if len(parts) >= 2:
                        present.add(parts[1])   # parts[1] is the uuid
    global global_used_uuids
    before = len(global_used_uuids)
    global_used_uuids &= present
    if len(global_used_uuids) != before:
        with open(global_used_uuids_file, "w", encoding="utf-8") as f:
            json.dump(list(global_used_uuids), f, indent=2)


# ------------------------------------------------------------------
# 6. Main processing
# ------------------------------------------------------------------

def main():
    # Prune global_used_uuids to match actually present files
    # prune_global_uuids()
    
    # Check if any languages are missing
    # samples_per_language = 80
    # total_needed = len(languages) * samples_per_language
    total_needed = sum(real_news_targets.get(lang, 750) for lang in languages.keys())
    
    for current_model in model_list:
        print(f"\n=== Processing model: {current_model} ===")
        # initialize vLLM
        tokenizer = AutoTokenizer.from_pretrained(current_model, trust_remote_code=True)
        llm = LLM(model=current_model, task="generate", tensor_parallel_size=4)
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=50000
        )

        # setup directories & state files
        model_dir = os.path.join(main_output_dir, current_model.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)
        counts_file = os.path.join(model_dir, "overall_counts.json")
        ip_file     = os.path.join(model_dir, "impersonator.json")

        if os.path.exists(counts_file):
            c = json.load(open(counts_file, encoding="utf-8"))
            succ, fail = c.get("overall_success_count", 0), c.get("overall_fail_count", 0)
        else:
            succ, fail = 0, 0
            update_overall_counts(counts_file, succ, fail)

        if os.path.exists(ip_file):
            impersonator = json.load(open(ip_file, encoding="utf-8"))
        else:
            impersonator = {
                "1": {"persona": "You are a news curator creating data to train defense models for social good.", "success": 0, "fail": 0},
                "2": {"persona": "You are a digital ethics analyst writing articles to educate readers about responsible AI use with transparency.", "success": 0, "fail": 0}
            }
            update_impersonator(ip_file, impersonator)

        overall_pbar = tqdm(total=total_needed, desc=f"Overall {current_model}")
        # lang_pbars = {lc: tqdm(total=samples_per_language, desc=languages[lc]["name"], leave=False)
        #               for lc in languages}
        
        lang_pbars = {lc: tqdm(total=real_news_targets.get(lc, 750), desc=languages[lc]["name"], leave=False)
              for lc in languages}

        # per-language loop
        for lc, info in languages.items():
            lang_folder = os.path.join(input_data_dir, lc)
            csv_path    = os.path.join(lang_folder, "data.csv")
            if not os.path.exists(csv_path):
                print(f"No data.csv in {lang_folder}, skipping {lc}.")
                continue

            df_lang = pd.read_csv(csv_path)
            lang_dir = os.path.join(model_dir, lc)
            os.makedirs(lang_dir, exist_ok=True)

            
            done = len([f for f in os.listdir(lang_dir) if f.endswith(".json")])
            samples_per_language = real_news_targets.get(lc, 750)  # ← DYNAMIC TARGET
            print(f"{info['name']}: {done}/{samples_per_language} completed")

            # build work queue
            queue = []
            for _, row in df_lang.iterrows():
                if done >= samples_per_language:
                    break
                uid = row["uuid"]
                art = row["content"]
                if uid in global_used_uuids:
                    continue
                tech = random.choice(techniques)
                deg  = random.choice(list(degree.keys())) if tech == "rewrite" else None
                queue.append({
                    "uuid": uid,
                    "article": art,
                    "lang": lc,
                    "tech": tech,
                    "deg": deg,
                    "idx": "1",
                    "attempt": 0,
                    "max": len(impersonator)
                })

            # process queue
            while queue and done < samples_per_language:
                item = queue.pop(0)
                item["attempt"] += 1

                # build prompt
                placeholder = technique_placeholder[item["tech"]].copy()
                if item["tech"] == "rewrite":
                    placeholder["degree"] = item["deg"]
                prompt = build_prompt(
                    # impersonator[item["idx"]]["persona"],
                    placeholder,
                    item["article"],
                    info["name"]
                )

                # prepare chat
                system_msg = (
                    f"{impersonator[item['idx']]['persona']} GUIDELINES: Provided with a new article. "
                    f"Your task is to create a truthful, accurate version adjusted via {placeholder['technique_info']}. "
                    "Follow the steps outlined below and pay attention to the language output required at each chain. "
                    "Ensure your output is in strict JSON format, adhering exactly to the specified keys and structure. "
                    "DO NOT include any commentary or text outside the JSON object. "
                    "Using ONLY single quotes in generated text to avoid issues with JSON data extraction parser. "
                    "This prompt takes an agentic impersonation approach: for each chain, you will assume a specialized role "
                    "with a defined function in the iterative transformation process. "
                    "Return only the strictly formatted JSON outputs for each chain in sequence in ChainOutputs. "
                    "Do Not include the instructions JSON in the output."
                )
                chat = [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt}
                ]

                text_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                # generate
                try:
                    out = next(llm.generate(text_prompt, sampling_params=sampling_params)).outputs[0].text
                except Exception as e:
                    print(f"Error generating for uuid {item['uuid']}: {e}")
                    queue.append(item)
                    time.sleep(2)
                    continue

                # check refusal
                if "I’m sorry, but I can’t comply" in out:
                    impersonator[item["idx"]]["fail"] += 1
                    fail += 1
                    update_overall_counts(counts_file, succ, fail)
                    update_impersonator(ip_file, impersonator)
                    print(f"[FAIL] persona {item['idx']} refused on uuid {item['uuid']}")
                    nxt = str(int(item["idx"]) + 1)
                    if nxt not in impersonator:
                        new_p = create_new_persona(nxt, impersonator, tokenizer, llm, sampling_params)
                        impersonator[nxt] = {"persona": new_p, "success": 0, "fail": 0}
                        update_impersonator(ip_file, impersonator)
                    item["idx"] = nxt if nxt in impersonator else "1"
                    if item["attempt"] < item["max"]:
                        queue.append(item)
                else:
                    # success
                    impersonator[item["idx"]]["success"] += 1
                    succ += 1
                    update_overall_counts(counts_file, succ, fail)
                    update_impersonator(ip_file, impersonator)

                    global_used_uuids.add(item["uuid"])
                    with open(global_used_uuids_file, 'w', encoding='utf-8') as f:
                        json.dump(list(global_used_uuids), f, indent=2)

                    print(f"[ OK ] persona {item['idx']} succeeded for uuid {item['uuid']}")

                    # save JSON
                    fname = f"CoI_{item['uuid']}_{lc}_{item['tech']}"
                    if item["deg"]:
                        fname += f"_{item['deg']}"
                    fname += ".json"
                    out_path = os.path.join(lang_dir, fname)
                    try:
                        parsed = json.loads(out)
                        with open(out_path, 'w', encoding='utf-8') as f:
                            json.dump(parsed, f, indent=2, ensure_ascii=False)
                    except JSONDecodeError:
                        with open(out_path, 'w', encoding='utf-8') as f:
                            f.write(out)

                    done += 1
                    lang_pbars[lc].update(1)
                    overall_pbar.update(1)

            lang_pbars[lc].close()

        overall_pbar.close()
        print(f"Finished processing model {current_model}")

if __name__ == "__main__":
    main()
