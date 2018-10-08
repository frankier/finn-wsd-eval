from stiff.munge.utils import space_tokenize
from stiff.utils.xml import iter_blocks


def iter_instances(inf):
    for lexelt in iter_blocks("lexelt")(inf):
        item = lexelt.get("item")
        pos = lexelt.get("pos")
        item_pos = "{}.{}".format(item, pos)
        for instance in lexelt.xpath("instance"):
            contexts = instance.xpath("context")
            assert len(contexts) == 1
            context_tag = contexts[0]
            heads = context_tag.xpath("head")
            assert len(heads) == 1
            head_tag = heads[0]
            head_text = head_tag.text
            before_texts = head_tag.xpath("preceding-sibling::text()")
            assert len(before_texts) == 1
            before_text = before_texts[0]
            after_texts = head_tag.xpath("following-sibling::text()")
            assert len(after_texts) == 1
            after_text = after_texts[0]
            inst_id = instance.attrib["id"]
            yield (
                inst_id,
                item_pos,
                (
                    space_tokenize(before_text),
                    space_tokenize(head_text),
                    space_tokenize(after_text)
                )
            )
