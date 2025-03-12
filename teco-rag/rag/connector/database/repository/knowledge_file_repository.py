# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.


from typing import Dict, List

from rag.connector.database.models.knowledge_base_model import KnowledgeBaseModel
from rag.connector.database.models.knowledge_file_model import FileDocModel, KnowledgeFileModel
from rag.connector.database.session import with_session
from rag.connector.database.utils import KnowledgeFile


@with_session
def delete_files_from_db(session, knowledge_base_name: str):
    # 删除文件
    session.query(KnowledgeFileModel).filter(
        KnowledgeFileModel.kb_name.ilike(knowledge_base_name)
    ).delete(synchronize_session=False)
    # 删除doc
    session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(knowledge_base_name)).delete(
        synchronize_session=False
    )
    # 删除关键词
    # session.query(KeywordsModel).filter(
    #     KeywordsModel.kb_name.ilike(knowledge_base_name)
    # ).delete(synchronize_session=False)

    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(knowledge_base_name))
        .first()
    )
    if kb:
        kb.file_count = 0
    session.commit()
    return True


@with_session
def list_files_from_db(session, knowledge_base_name: str):
    files = (
        session.query(KnowledgeFileModel)
        .filter(KnowledgeFileModel.kb_name.ilike(knowledge_base_name))
        .all()
    )
    file_names = [f.file_name for f in files]
    return file_names


@with_session
def file_info_from_db(session, file_name: str):
    """
    从数据库中获取指定文件名的文件信息。

    Args:
        session (Session): 数据库会话对象。
        file_name (str): 要查询的文件名。

    Returns:
        KnowledgeFileModel | None: 如果找到匹配的文件，返回KnowledgeFileModel对象；否则返回None。
    """
    file = (
        session.query(KnowledgeFileModel)
        .filter(KnowledgeFileModel.file_name.ilike(file_name))
        .first()
    )
    return file


@with_session
def list_docs_from_db(
    session,
    kb_name: str,
    file_name: str = None,
    metadata: Dict = {},
) -> List[Dict]:
    """
    列出某知识库某文件对应的所有Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    """
    docs = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))
    if file_name:
        docs = docs.filter(FileDocModel.file_name.ilike(file_name))
    for k, v in metadata.items():
        docs = docs.filter(FileDocModel.meta_data[k].as_string() == str(v))

    return [{"id": x.doc_id, "metadata": x.metadata} for x in docs.all()]


@with_session
def delete_docs_from_db(
    session,
    kb_name: str,
    file_name: str = None,
) -> List[Dict]:
    """
    删除某知识库某文件对应的所有Document，并返回被删除的Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    """
    docs = list_docs_from_db(kb_name=kb_name, file_name=file_name)
    doc_ids = [doc["id"] for doc in docs]

    session.query(FileDocModel).filter(
        FileDocModel.kb_name == kb_name, FileDocModel.doc_id.in_(doc_ids)
    ).delete(synchronize_session=False)
    # session.query(KeywordsModel).filter(KeywordsModel.doc_id.in_(doc_ids)).delete(
    #     synchronize_session=False
    # )

    return docs


@with_session
def delete_file_from_db(session, kb_file: KnowledgeFile):
    """从数据库中删除指定的知识文件及其相关文档

    Args:
        session: 数据库会话对象
        kb_file (KnowledgeFile): 要删除的知识文件对象

    Returns:
        bool: 删除是否成功,总是返回True
    """
    # 查询是否存在该文件
    existing_file = (
        session.query(KnowledgeFileModel)
        .filter(
            KnowledgeFileModel.file_name.ilike(kb_file.filename),
            KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
        )
        .first()
    )

    if existing_file:
        # 删除文件记录
        session.delete(existing_file)
        # 删除文件关联的所有文档
        delete_docs_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        session.commit()

        # 更新知识库的文件计数
        kb = (
            session.query(KnowledgeBaseModel)
            .filter(KnowledgeBaseModel.kb_name.ilike(kb_file.kb_name))
            .first()
        )
        if kb:
            kb.file_count -= 1
            session.commit()
    return True


@with_session
def add_file_to_db(
    session,
    kb_file: KnowledgeFile,
    docs_count: int = 0,
    custom_docs: bool = False,
):
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_file.kb_name).first()
    if kb:
        # 如果已经存在该文件，则更新文件信息与版本号
        existing_file: KnowledgeFileModel = (
            session.query(KnowledgeFileModel)
            .filter(
                KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
                KnowledgeFileModel.file_name.ilike(kb_file.filename),
            )
            .first()
        )
        mtime = kb_file.get_mtime()
        size = kb_file.get_size()

        if existing_file:
            existing_file.file_mtime = mtime
            existing_file.file_size = size
            existing_file.docs_count = docs_count
            existing_file.custom_docs = custom_docs
            existing_file.file_version += 1
        # 否则，添加新文件
        else:
            new_file = KnowledgeFileModel(
                file_name=kb_file.filename,
                file_ext=kb_file.ext,
                kb_name=kb_file.kb_name,
                document_loader_name=kb_file.document_loader.__name__,
                text_splitter_name=kb_file.text_splitter.__name__,
                file_mtime=mtime,
                file_size=size,
                docs_count=docs_count,
                custom_docs=custom_docs,
            )
            kb.file_count += 1
            session.add(new_file)
    return True


@with_session
def add_docs_to_db(session, kb_name: str, file_name: str, doc_infos: List[Dict]):
    """
    将某知识库某文件对应的所有Document信息添加到数据库。
    doc_infos形式：[{"id": str, "metadata": dict, "page_content": str}, ...]
    """
    #! 这里会出现doc_infos为None的情况，需要进一步排查
    if doc_infos is None:
        print(
            "输入的server.db.repository.knowledge_file_repository.add_docs_to_db的doc_infos参数为None"
        )
        return False
    for d in doc_infos:
        obj = FileDocModel(
            kb_name=kb_name,
            file_name=file_name,
            doc_id=d["id"],
            meta_data=d["metadata"],
            page_content=d["page_content"],
        )
        session.add(obj)
    return True


@with_session
def get_file_detail(session, kb_name: str, filename: str) -> dict:
    file: KnowledgeFileModel = (
        session.query(KnowledgeFileModel)
        .filter(
            KnowledgeFileModel.file_name.ilike(filename),
            KnowledgeFileModel.kb_name.ilike(kb_name),
        )
        .first()
    )
    if file:
        return {
            "kb_name": file.kb_name,
            "file_name": file.file_name,
            "file_ext": file.file_ext,
            "file_version": file.file_version,
            "document_loader": file.document_loader_name,
            "text_splitter": file.text_splitter_name,
            "create_time": file.create_time,
            "file_mtime": file.file_mtime,
            "file_size": file.file_size,
            "custom_docs": file.custom_docs,
            "docs_count": file.docs_count,
        }
    else:
        return {}
